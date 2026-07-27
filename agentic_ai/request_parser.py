"""Small deterministic parser for the first agent workflow version."""

import re


def add_once(items, value):
    if value not in items:
        items.append(value)


def extract_budget(request):
    patterns = [
        r"(?:under|below|within|up to|budget(?: of)?(?: is)?)\s*"
        r"(?:₹|rs\.?|inr)?\s*([\d,]+)",
        r"(?:₹|rs\.?|inr)\s*([\d,]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, request, flags=re.IGNORECASE)
        if match:
            return float(match.group(1).replace(",", ""))
    return None


def parse_user_request(request, existing_preferences, catalog):
    """Merge request details with known profile and order-history values."""
    if not str(request).strip():
        raise ValueError("user_request cannot be blank")

    preferences = dict(existing_preferences or {})
    text = " ".join(str(request).casefold().replace("-", " ").split())

    city, locality = catalog.find_location(request)
    if city:
        preferences["city"] = city
    if locality:
        preferences["locality"] = locality

    for meal in ("breakfast", "lunch", "dinner", "snack"):
        if meal in text:
            preferences["meal_time"] = meal.title()
            break

    if "non vegetarian" in text or "non veg" in text:
        preferences["dietary_preference"] = "Non-veg"
    elif "vegan" in text:
        preferences["dietary_preference"] = "Veg"
    elif "vegetarian" in text or re.search(r"\bveg\b", text):
        preferences["dietary_preference"] = "Veg"

    for word, value in (
        ("spicy", "Hot"), ("hot", "Hot"),
        ("medium spice", "Medium"), ("mild", "Mild"),
    ):
        if word in text:
            preferences["spice_preference"] = value
            break

    budget = extract_budget(request)
    if budget is not None:
        preferences["user_budget"] = budget

    rating_match = re.search(r"(?:rating|rated)\s*(?:of|above|at least)?\s*([0-5](?:\.\d)?)", text)
    if rating_match:
        preferences["minimum_rating"] = float(rating_match.group(1))

    restrictions = list(preferences.get("restrictions", []))
    if "lactose intolerant" in text or "lactose intolerance" in text:
        add_once(restrictions, "Lactose intolerance")
    if "gluten sensitive" in text or "gluten sensitivity" in text:
        add_once(restrictions, "Gluten sensitivity")
    if "vegan" in text:
        add_once(restrictions, "Vegan")
    elif preferences.get("dietary_preference") == "Veg":
        add_once(restrictions, "Vegetarian")

    if "allerg" in text or "free" in text:
        allergy_terms = {
            "peanut": "Peanut allergy",
            "groundnut": "Peanut allergy",
            "tree nut": "Tree-nut allergy",
            "cashew": "Tree-nut allergy",
            "almond": "Tree-nut allergy",
            "egg": "Egg allergy",
            "soy": "Soy allergy",
            "soya": "Soy allergy",
        }
        for term, restriction in allergy_terms.items():
            if term in text:
                add_once(restrictions, restriction)

    preferences["restrictions"] = restrictions
    return preferences

