"""LangChain tools used by the controlled culinary workflow."""

from functools import lru_cache

from langchain.tools import tool

from rag import CulinaryRAG
from src.dish_filter import DishCatalog
from src.ml_prediction import CuisinePredictor
from src.safety_rules import screen_dishes
from src.spending_prediction import SpendingPredictor


@lru_cache(maxsize=1)
def get_catalog():
    return DishCatalog()


@lru_cache(maxsize=1)
def get_cuisine_predictor():
    return CuisinePredictor()


@lru_cache(maxsize=1)
def get_spending_predictor():
    return SpendingPredictor()


@lru_cache(maxsize=1)
def get_rag():
    return CulinaryRAG()


@tool
def predict_cuisines(user_features: dict) -> list[dict]:
    """Return all cuisine probabilities for one returning user's feature record."""
    return get_cuisine_predictor().predict_probabilities(user_features)


@tool
def predict_spending_limit(user_context: dict) -> int:
    """Predict a spending limit only when the user did not provide a budget."""
    return get_spending_predictor().predict_spending_limit(user_context)


@tool
def filter_restaurant_dishes(
    city: str,
    locality: str | None,
    cuisine: str,
    budget: float,
    dietary_preference: str | None = None,
    minimum_rating: float | None = None,
    top_n: int = 10,
) -> list[dict]:
    """Return real restaurant dishes matching all supplied hard constraints."""
    return get_catalog().filter(
        city=city,
        locality=locality,
        cuisine=cuisine,
        budget=budget,
        dietary_preference=dietary_preference,
        minimum_rating=minimum_rating,
        top_n=top_n,
    )


@tool
def retrieve_dish_knowledge(candidates: list[dict]) -> list[dict]:
    """Retrieve structured culinary knowledge for shortlisted restaurant dishes."""
    return [result.to_dict() for result in get_rag().retrieve_many(candidates)]


@tool
def apply_safety_rules(restrictions: list[str], rag_results: list[dict]) -> list[dict]:
    """Apply deterministic dietary and allergy rules to every RAG result."""
    return screen_dishes(restrictions, rag_results)

