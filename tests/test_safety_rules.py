"""Focused checks for deterministic dietary and allergen rules."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.safety_rules import check_dish_safety, screen_dishes


def rag_result(
    dish_name,
    *,
    ingredients="",
    allergens="none_identified_from_name",
    tags="vegetarian",
    variations="",
    preparation="",
    status="matched",
):
    result = {
        "requested_dish": dish_name,
        "matched_dish_id": "test-dish" if status == "matched" else None,
        "match_type": "exact" if status == "matched" else "none",
        "common_ingredients": ingredients,
        "allergens": allergens,
        "dietary_tags": tags,
        "ingredient_variations": variations,
        "preparation_method": preparation,
    }
    if status != "matched":
        result["retrieval_status"] = status
    return result


def main():
    paneer = check_dish_safety(
        ["Lactose intolerant", "Vegetarian"],
        rag_result(
            "Paneer Butter Masala",
            ingredients="paneer;butter;cream;tomato",
            allergens="dairy;possible_tree_nuts",
        ),
    )
    assert paneer["safety_status"] == "Rejected due to known conflict"
    assert {"paneer", "butter", "cream", "dairy"} <= set(paneer["detected_conflicts"])

    dal = check_dish_safety(
        "Lactose intolerance",
        rag_result("Dal Tadka", ingredients="lentils;tomato", variations="may use ghee"),
    )
    assert dal["safety_status"] == "Possible dietary conflict"
    assert dal["detected_conflicts"] == ["ghee"]

    chana = check_dish_safety(
        ["Lactose intolerance", "Vegetarian"],
        rag_result("Chana Masala", ingredients="chickpeas;tomato;onion;spices"),
    )
    assert chana["safety_status"] == "Likely compatible"
    assert "cannot guarantee medical safety" in chana["disclaimer"]

    peanut = check_dish_safety(
        "Peanut allergy",
        rag_result("Peanut Chutney", ingredients="peanut;chilli", allergens="peanut"),
    )
    assert peanut["safety_status"] == "Rejected due to known conflict"

    traces = check_dish_safety(
        "Peanut allergy",
        rag_result("Mixed Chutney", ingredients="coconut;chilli", allergens="may contain traces of peanuts"),
    )
    assert traces["safety_status"] == "Possible dietary conflict"

    unavailable = check_dish_safety(
        "Soy allergy",
        rag_result("Royal Chef Bowl", status="no_reliable_match"),
    )
    assert unavailable["safety_status"] == "Ingredient information unavailable"
    assert unavailable["conflict_level"] == "Unknown"

    vegetarian_paneer = check_dish_safety(
        "Vegetarian",
        rag_result("Paneer Tikka", ingredients="paneer;spices", allergens="dairy"),
    )
    vegan_paneer = check_dish_safety(
        "Vegan",
        rag_result("Paneer Tikka", ingredients="paneer;spices", allergens="dairy"),
    )
    assert vegetarian_paneer["safety_status"] == "Likely compatible"
    assert vegan_paneer["safety_status"] == "Rejected due to known conflict"

    eggplant = check_dish_safety(
        "Egg allergy",
        rag_result("Eggplant Curry", ingredients="eggplant;tomato;spices"),
    )
    assert eggplant["safety_status"] == "Likely compatible"

    direct_cases = [
        ("Tree-nut allergy", "Kaju Curry", "cashew;tomato", "cashew"),
        ("Gluten sensitivity", "Naan", "wheat flour;salt", "wheat"),
        ("Egg allergy", "Egg Curry", "eggs;tomato", "eggs"),
        ("Soy allergy", "Tofu Stir Fry", "tofu;vegetables", "tofu"),
        ("Vegetarian", "Chicken Curry", "chicken;spices", "chicken"),
    ]
    for restriction, dish, ingredients, expected_term in direct_cases:
        result = check_dish_safety(
            restriction,
            rag_result(dish, ingredients=ingredients),
        )
        assert result["safety_status"] == "Rejected due to known conflict"
        assert expected_term in result["detected_conflicts"]

    screened = screen_dishes("Lactose intolerance", [
        rag_result("Paneer Curry", ingredients="paneer;tomato", allergens="dairy"),
        rag_result("Chana Masala", ingredients="chickpeas;tomato"),
    ])
    assert [item["safety_status"] for item in screened] == [
        "Rejected due to known conflict",
        "Likely compatible",
    ]

    print("All safety rule checks passed.")


if __name__ == "__main__":
    main()
