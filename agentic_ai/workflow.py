"""Simple LangChain tool workflow for culinary recommendations."""

from LLM import generate_recommendation, prepare_llm_context
from src.ml_prediction import MINIMUM_HISTORY_ORDERS
from .request_parser import parse_user_request
from .tools import (
    apply_safety_rules,
    filter_restaurant_dishes,
    get_catalog,
    predict_cuisines,
    retrieve_dish_knowledge,
)


NO_MATCH_MESSAGE = (
    "No suitable dish was found for your current location, budget, cuisine "
    "preference, and dietary restrictions. Try changing the cuisine or "
    "increasing the search range. Dietary restrictions were not relaxed."
)


def classifier_input(preferences):
    features = dict(preferences.get("classifier_features", {}))
    if preferences.get("meal_time"):
        features["meal_time"] = preferences["meal_time"]
    if preferences.get("spice_preference"):
        features["spice_preference"] = preferences["spice_preference"]
    if preferences.get("dietary_preference"):
        features["vegetarian"] = (
            "Yes" if preferences["dietary_preference"] == "Veg" else "No"
        )
    return features


def filter_dishes(preferences, budget, cuisine):
    return filter_restaurant_dishes.invoke({
        "city": preferences["city"],
        "locality": preferences.get("locality"),
        "cuisine": cuisine,
        "budget": budget,
        "dietary_preference": preferences.get("dietary_preference"),
        "top_n": 10,
    })


def no_match(state):
    state["final_response"] = NO_MATCH_MESSAGE
    return state


def run_workflow(user_request, user_profile=None):
    """Run the culinary tools in their required safety order."""
    preferences = parse_user_request(user_request, user_profile, get_catalog())
    budget = float(preferences["user_budget"])

    features = classifier_input(preferences)
    if int(features.get("order_frequency", 0)) >= MINIMUM_HISTORY_ORDERS:
        predictions = predict_cuisines.invoke({"user_features": features})
        prediction_source = "history_model"
    else:
        preferred = preferences.get("preferred_cuisines", [])
        predictions = [
            {"cuisine": str(cuisine), "probability": None}
            for cuisine in preferred
        ]
        prediction_source = "onboarding"

    primary = predictions[0]
    secondary = predictions[1] if len(predictions) > 1 else None
    predicted_cuisine = {
        "primary": primary,
        "secondary": secondary,
        "selected": primary,
        "predictions": predictions[:3],
        "source": prediction_source,
    }

    state = {
        "user_request": user_request,
        "parsed_preferences": preferences,
        "predicted_cuisine": predicted_cuisine,
        "final_budget": budget,
        "filtered_candidates": [],
        "rag_results": [],
        "safety_results": [],
        "final_validated_candidates": [],
        "llm_context": {},
        "final_response": "",
    }

    candidates = filter_dishes(preferences, budget, primary["cuisine"])
    if not candidates and secondary:
        predicted_cuisine["selected"] = secondary
        candidates = filter_dishes(preferences, budget, secondary["cuisine"])

    state["filtered_candidates"] = candidates
    if not candidates:
        return no_match(state)

    rag_results = retrieve_dish_knowledge.invoke({"candidates": candidates})
    safety_results = apply_safety_rules.invoke({
        "restrictions": preferences.get("restrictions", []),
        "rag_results": rag_results,
    })
    state["rag_results"] = rag_results
    state["safety_results"] = safety_results

    validated = []
    for candidate, knowledge, safety in zip(candidates, rag_results, safety_results):
        status = safety["safety_status"]
        if status not in {"Likely compatible", "Possible dietary conflict"}:
            continue

        validated.append({
            "dish_name": candidate["dish_name"],
            "restaurant_name": candidate["restaurant_name"],
            "locality": candidate["locality"],
            "price": candidate["dish_price"],
            "restaurant_rating": candidate["restaurant_rating"],
            "common_ingredients": knowledge.get("common_ingredients"),
            "allergens": knowledge.get("allergens"),
            "possible_substitutions": knowledge.get("possible_substitutions"),
            "safety_status": status,
            "reason": safety["reason"],
        })

    if not validated:
        return no_match(state)

    safety_order = {"Likely compatible": 0, "Possible dietary conflict": 1}
    validated.sort(key=lambda dish: (
        safety_order[dish["safety_status"]],
        -(float(dish["restaurant_rating"]) if dish["restaurant_rating"] else 0),
        float(dish["price"]),
    ))
    state["final_validated_candidates"] = [
        {"rank": rank, **dish} for rank, dish in enumerate(validated, start=1)
    ]
    state["llm_context"] = prepare_llm_context(state)

    state["final_response"] = generate_recommendation(state["llm_context"])
    return state
