"""Controlled LangGraph workflow for structured culinary recommendations."""

from typing import Any, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from LLM import generate_recommendation, prepare_llm_context
from src.ml_prediction import MINIMUM_HISTORY_ORDERS
from .request_parser import parse_user_request
from .tools import (
    apply_safety_rules,
    filter_restaurant_dishes,
    get_catalog,
    predict_cuisines,
    predict_spending_limit,
    retrieve_dish_knowledge,
)


NO_MATCH_MESSAGE = (
    "No suitable dish was found for your current location, budget, cuisine "
    "preference, and dietary restrictions. Try changing the cuisine or "
    "increasing the search range. Dietary restrictions were not relaxed."
)


class WorkflowState(TypedDict):
    user_request: str
    parsed_preferences: NotRequired[dict[str, Any]]
    predicted_cuisine: NotRequired[dict[str, Any]]
    final_budget: NotRequired[float]
    filtered_candidates: NotRequired[list[dict]]
    rag_results: NotRequired[list[dict]]
    safety_results: NotRequired[list[dict]]
    final_validated_candidates: NotRequired[list[dict]]
    error_message: NotRequired[str]
    llm_context: NotRequired[dict[str, Any]]
    final_response: NotRequired[str]


def parse_request_node(state):
    preferences = parse_user_request(
        state["user_request"], state.get("parsed_preferences"), get_catalog()
    )
    if not preferences.get("city"):
        raise ValueError("The request must provide a supported city or locality")
    return {"parsed_preferences": preferences, "error_message": ""}


def classifier_input(preferences):
    features = dict(preferences.get("classifier_features", {}))
    feature_names = (
        "meal_time", "day_type", "vegetarian",
        "average_previous_rating", "most_ordered_cuisine", "order_frequency",
        "spice_preference", "previous_cuisine_orders",
    )
    for name in feature_names:
        if name in preferences:
            features[name] = preferences[name]

    if preferences.get("meal_time"):
        features["meal_time"] = preferences["meal_time"]
    if preferences.get("spice_preference"):
        features["spice_preference"] = preferences["spice_preference"]
    if preferences.get("dietary_preference"):
        features["vegetarian"] = (
            "Yes" if preferences["dietary_preference"] == "Veg" else "No"
        )
    return features


def predict_cuisine_node(state):
    preferences = state["parsed_preferences"]
    features = classifier_input(preferences)

    if int(features.get("order_frequency", 0)) >= MINIMUM_HISTORY_ORDERS:
        predictions = predict_cuisines.invoke({"user_features": features})
        prediction_source = "history_model"
    else:
        preferred = preferences.get("preferred_cuisines", [])
        if isinstance(preferred, str):
            preferred = [preferred]
        predictions = [
            {"cuisine": str(cuisine), "probability": None} for cuisine in preferred
        ]
        if not predictions:
            raise ValueError(
                "Cuisine prediction needs returning-user history or onboarding cuisines"
            )
        prediction_source = "onboarding"

    primary = predictions[0]
    secondary = predictions[1] if len(predictions) > 1 else None
    return {
        "predicted_cuisine": {
            "primary": primary,
            "secondary": secondary,
            "selected": primary,
            "predictions": predictions[:3],
            "source": prediction_source,
        }
    }


def budget_route(state):
    preferences = state["parsed_preferences"]
    if preferences.get("user_budget") is not None:
        return "user"
    if preferences.get("spending_features"):
        return "predict"
    raise ValueError("New users must provide an explicit budget")


def use_user_budget_node(state):
    budget = float(state["parsed_preferences"]["user_budget"])
    if budget <= 0:
        raise ValueError("User budget must be positive")
    return {"final_budget": budget}


def spending_input(state):
    preferences = state["parsed_preferences"]
    context = dict(preferences.get("spending_features", {}))
    feature_names = (
        "user_average_order_value", "meal_time", "weekday_or_weekend",
        "location", "payment_method", "previous_order_count", "preferred_cuisine",
    )
    for name in feature_names:
        if name in preferences:
            context[name] = preferences[name]

    context["meal_time"] = preferences.get("meal_time", context.get("meal_time"))
    context["location"] = preferences["city"]
    context["preferred_cuisine"] = state["predicted_cuisine"]["primary"]["cuisine"]
    return context


def predict_budget_node(state):
    budget = predict_spending_limit.invoke({"user_context": spending_input(state)})
    return {"final_budget": float(budget)}


def filter_for_cuisine(state, cuisine):
    preferences = state["parsed_preferences"]
    return filter_restaurant_dishes.invoke({
        "city": preferences["city"],
        "locality": preferences.get("locality"),
        "cuisine": cuisine,
        "budget": state["final_budget"],
        "dietary_preference": preferences.get("dietary_preference"),
        "minimum_rating": preferences.get("minimum_rating"),
        "top_n": 10,
    })


def filter_dishes_node(state):
    cuisine = state["predicted_cuisine"]["primary"]["cuisine"]
    return {"filtered_candidates": filter_for_cuisine(state, cuisine)}


def no_candidate_check(state):
    return "continue" if state.get("filtered_candidates") else "fallback"


def try_second_cuisine_node(state):
    prediction = dict(state["predicted_cuisine"])
    secondary = prediction.get("secondary")
    if not secondary:
        return {
            "filtered_candidates": [],
            "error_message": NO_MATCH_MESSAGE,
        }

    prediction["selected"] = secondary
    candidates = filter_for_cuisine(state, secondary["cuisine"])
    message = (
        f"Primary cuisine had no matches; used second predicted cuisine: "
        f"{secondary['cuisine']}."
        if candidates
        else NO_MATCH_MESSAGE
    )
    return {
        "predicted_cuisine": prediction,
        "filtered_candidates": candidates,
        "error_message": message,
    }


def fallback_check(state):
    return "continue" if state.get("filtered_candidates") else "stop"


def retrieve_ingredients_node(state):
    results = retrieve_dish_knowledge.invoke({
        "candidates": state["filtered_candidates"]
    })
    return {"rag_results": results}


def check_restrictions_node(state):
    restrictions = state["parsed_preferences"].get("restrictions", [])
    results = apply_safety_rules.invoke({
        "restrictions": restrictions,
        "rag_results": state["rag_results"],
    })
    return {"safety_results": results}


def select_candidates_node(state):
    selected_cuisine = state["predicted_cuisine"]["selected"]
    validated = []

    for candidate, knowledge, safety in zip(
        state["filtered_candidates"], state["rag_results"], state["safety_results"]
    ):
        status = safety["safety_status"]
        if status not in {"Likely compatible", "Possible dietary conflict"}:
            continue

        validated.append({
            "recommendation_type": "primary" if status == "Likely compatible" else "warning",
            "dish_name": candidate["dish_name"],
            "restaurant_name": candidate["restaurant_name"],
            "city": candidate["city"],
            "locality": candidate["locality"],
            "price": candidate["dish_price"],
            "restaurant_rating": candidate["restaurant_rating"],
            "rating_count": candidate["rating_count"],
            "predicted_cuisine": selected_cuisine["cuisine"],
            "cuisine_probability": selected_cuisine.get("probability"),
            "matched_dish": knowledge.get("matched_dish"),
            "common_ingredients": knowledge.get("common_ingredients"),
            "allergens": knowledge.get("allergens"),
            "dietary_tags": knowledge.get("dietary_tags"),
            "possible_substitutions": knowledge.get("possible_substitutions"),
            "source": knowledge.get("source"),
            "safety_status": status,
            "detected_conflicts": safety["detected_conflicts"],
            "conflict_level": safety["conflict_level"],
            "reason": safety["reason"],
            "disclaimer": safety["disclaimer"],
        })

    return {"final_validated_candidates": validated}


def validated_candidate_check(state):
    return "continue" if state.get("final_validated_candidates") else "no_match"


def no_match_response_node(state):
    """Return a fixed response without sending empty results to an LLM."""
    return {
        "final_validated_candidates": [],
        "error_message": NO_MATCH_MESSAGE,
        "final_response": NO_MATCH_MESSAGE,
    }


def rank_candidates_node(state):
    safety_order = {"Likely compatible": 0, "Possible dietary conflict": 1}
    ranked = sorted(
        state.get("final_validated_candidates", []),
        key=lambda item: (
            safety_order[item["safety_status"]],
            -(float(item["restaurant_rating"]) if item["restaurant_rating"] else 0),
            float(item["price"]),
        ),
    )
    return {"final_validated_candidates": ranked}


def prepare_final_context_node(state):
    candidates = []
    for rank, candidate in enumerate(state.get("final_validated_candidates", []), start=1):
        candidates.append({"rank": rank, **candidate})

    updated_state = {
        "final_validated_candidates": candidates,
        "error_message": state.get("error_message", ""),
        "final_response": "",
    }
    return {**updated_state, "llm_context": prepare_llm_context({**state, **updated_state})}


def generate_response_node(state, generator=generate_recommendation):
    return {"final_response": generator(state["llm_context"])}


def build_workflow(llm_generator=None):
    """Build and compile the deterministic culinary recommendation graph."""
    generator = llm_generator or generate_recommendation

    def call_llm(state):
        return generate_response_node(state, generator)

    graph = StateGraph(WorkflowState)
    graph.add_node("parse_request", parse_request_node)
    graph.add_node("predict_cuisine", predict_cuisine_node)
    graph.add_node("use_user_budget", use_user_budget_node)
    graph.add_node("predict_budget", predict_budget_node)
    graph.add_node("filter_dishes", filter_dishes_node)
    graph.add_node("try_second_cuisine", try_second_cuisine_node)
    graph.add_node("retrieve_ingredients", retrieve_ingredients_node)
    graph.add_node("check_restrictions", check_restrictions_node)
    graph.add_node("select_candidates", select_candidates_node)
    graph.add_node("no_match_response", no_match_response_node)
    graph.add_node("rank_candidates", rank_candidates_node)
    graph.add_node("prepare_final_context", prepare_final_context_node)
    graph.add_node("generate_response", call_llm)

    graph.add_edge(START, "parse_request")
    graph.add_edge("parse_request", "predict_cuisine")
    graph.add_conditional_edges(
        "predict_cuisine", budget_route,
        {"user": "use_user_budget", "predict": "predict_budget"},
    )
    graph.add_edge("use_user_budget", "filter_dishes")
    graph.add_edge("predict_budget", "filter_dishes")
    graph.add_conditional_edges(
        "filter_dishes", no_candidate_check,
        {"continue": "retrieve_ingredients", "fallback": "try_second_cuisine"},
    )
    graph.add_conditional_edges(
        "try_second_cuisine", fallback_check,
        {"continue": "retrieve_ingredients", "stop": "no_match_response"},
    )
    graph.add_edge("retrieve_ingredients", "check_restrictions")
    graph.add_edge("check_restrictions", "select_candidates")
    graph.add_conditional_edges(
        "select_candidates", validated_candidate_check,
        {"continue": "rank_candidates", "no_match": "no_match_response"},
    )
    graph.add_edge("rank_candidates", "prepare_final_context")
    graph.add_edge("no_match_response", END)
    graph.add_edge("prepare_final_context", "generate_response")
    graph.add_edge("generate_response", END)
    return graph.compile()


def run_workflow(user_request, user_profile=None, llm_generator=None):
    """Run the graph and return its structured shared state."""
    initial_state = {
        "user_request": user_request,
        "parsed_preferences": dict(user_profile or {}),
        "predicted_cuisine": {},
        "filtered_candidates": [],
        "rag_results": [],
        "safety_results": [],
        "final_validated_candidates": [],
        "error_message": "",
        "llm_context": {},
        "final_response": "",
    }
    return build_workflow(llm_generator).invoke(initial_state)
