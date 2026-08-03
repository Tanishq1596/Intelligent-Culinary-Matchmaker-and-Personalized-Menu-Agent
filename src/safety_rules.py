"""Deterministic dietary and allergen checks for retrieved dish knowledge."""

import re


LIKELY_COMPATIBLE = "Likely compatible"
POSSIBLE_CONFLICT = "Possible dietary conflict"
REJECTED = "Rejected due to known conflict"
INFORMATION_UNAVAILABLE = "Ingredient information unavailable"

DISCLAIMER = (
    "Generic recipe data cannot guarantee medical safety. Confirm ingredients "
    "and cross-contamination risks directly with the restaurant."
)

# Each restriction maps to ingredients or tags that conflict with it.
CONFLICT_TERMS = {
    "lactose intolerance": (
        "dairy", "milk", "cream", "butter", "paneer", "cheese",
        "curd", "yogurt", "yoghurt", "ghee", "khoya",
    ),
    "peanut allergy": ("peanut", "peanuts", "groundnut", "groundnuts"),
    "tree nut allergy": (
        "tree nut", "tree nuts", "cashew", "cashews", "almond", "almonds",
        "walnut", "walnuts", "pistachio", "pistachios", "hazelnut", "hazelnuts",
    ),
    "gluten sensitivity": (
        "gluten", "wheat", "maida", "semolina", "sooji", "suji", "barley", "bread",
    ),
    "egg allergy": ("egg", "eggs", "mayonnaise", "mayo"),
    "soy allergy": ("soy", "soya", "soybean", "soybeans", "tofu", "tempeh"),
    "vegan": (
        "dairy", "milk", "cream", "butter", "paneer", "cheese", "curd",
        "yogurt", "yoghurt", "ghee", "khoya", "egg", "eggs", "mayonnaise",
        "chicken", "mutton", "lamb", "fish", "seafood", "meat", "beef", "pork",
        "prawn", "prawns", "shrimp", "gelatin", "honey", "non vegetarian",
        "non veg", "contains egg",
    ),
    "vegetarian": (
        "egg", "eggs", "chicken", "mutton", "lamb", "fish", "seafood", "meat",
        "beef", "pork", "prawn", "prawns", "shrimp", "gelatin",
        "non vegetarian", "non veg", "contains egg",
    ),
}

DIRECT_FIELDS = ("common_ingredients", "allergens", "dietary_tags")
POSSIBLE_FIELDS = ("ingredient_variations", "preparation_method")
UNCERTAIN_WORDS = (
    "possible", "possibly", "may contain", "may use", "might contain",
    "trace", "traces", "cross contamination", "varies", "variation",
)


def normalize_text(value):
    """Create lowercase searchable text while preserving word boundaries."""
    text = str(value or "").casefold().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def has_term(text, term):
    """Match a whole ingredient term, so 'egg' does not match 'eggplant'."""
    return f" {normalize_text(term)} " in f" {normalize_text(text)} "


def find_conflicts(record, restrictions):
    """Return detected terms with direct conflicts taking priority."""
    conflicts = {}

    for restriction in restrictions:
        for field in DIRECT_FIELDS + POSSIBLE_FIELDS:
            default_level = "possible" if field in POSSIBLE_FIELDS else "direct"
            for item in record.get(field, "").split(","):
                item = normalize_text(item)
                level = default_level
                if any(has_term(item, word) for word in UNCERTAIN_WORDS):
                    level = "possible"

                for term in CONFLICT_TERMS[restriction]:
                    if not has_term(item, term):
                        continue
                    previous_level = conflicts.get(term)
                    if previous_level != "direct":
                        conflicts[term] = level

    direct = [term for term, level in conflicts.items() if level == "direct"]
    possible = [term for term, level in conflicts.items() if level == "possible"]
    return direct, possible


def check_dish_safety(user_restrictions, rag_result):
    """Classify one RAG result using deterministic dietary and allergy rules."""
    restrictions = user_restrictions or []
    record = rag_result
    dish_name = (
        record.get("requested_dish")
        or record.get("dish_name")
        or record.get("matched_dish")
        or "Unknown dish"
    )

    if not record.get("matched_dish_id"):
        return {
            "dish_name": dish_name,
            "safety_status": INFORMATION_UNAVAILABLE,
            "detected_conflicts": [],
            "conflict_level": "Unknown",
            "reason": "RAG did not return a reliable match, so compatibility cannot be assessed.",
            "disclaimer": DISCLAIMER,
        }

    direct, possible = find_conflicts(record, restrictions)

    if direct:
        status = REJECTED
        level = "Direct"
        detected = direct + [term for term in possible if term not in direct]
        reason = "Known restricted terms were found: " + ", ".join(direct) + "."
    elif possible:
        status = POSSIBLE_CONFLICT
        level = "Possible"
        detected = possible
        reason = "Restricted terms may occur depending on preparation: " + ", ".join(possible) + "."
    else:
        status = LIKELY_COMPATIBLE
        level = "None"
        detected = []
        reason = "No listed conflicts were found in the matched generic recipe information."

    return {
        "dish_name": dish_name,
        "safety_status": status,
        "detected_conflicts": detected,
        "conflict_level": level,
        "reason": reason,
        "disclaimer": DISCLAIMER,
    }


def screen_dishes(user_restrictions, rag_results):
    """Apply the same safety rules to every shortlisted RAG result."""
    return [check_dish_safety(user_restrictions, result) for result in rag_results]
