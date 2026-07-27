"""Generate a grounded recommendation from validated dish candidates."""

import json
import os

from google import genai
from google.genai import types


MODEL_NAME = "gemini-3.6-flash"

SYSTEM_INSTRUCTIONS = """
You explain validated culinary recommendations.

Rules:
- Use only the candidate dishes supplied in the context.
- Preserve dish names, restaurant names, prices, ingredients, and substitutions.
- Recommend likely compatible dishes before possible dietary conflicts.
- Clearly explain every possible dietary conflict.
- Never recommend rejected dishes or dishes with unavailable ingredient information.
- Do not invent dishes, restaurants, prices, ingredients, or substitutions.
- Never claim that a dish is guaranteed safe, allergy-safe, or clinically safe.
- State that restaurant ingredients and preparation may vary.

Use this response structure:
Best recommendation
Why it matches
Restaurant and price
Dietary note
Alternative recommendation
Preparation disclaimer
""".strip()


def prepare_llm_context(state):
    """Keep only validated facts needed for the final explanation."""
    preferences = state["parsed_preferences"]
    selected_cuisine = state["predicted_cuisine"]["selected"]["cuisine"]

    candidates = [
        {
            "dish_name": dish["dish_name"],
            "restaurant_name": dish["restaurant_name"],
            "price": dish["price"],
            "common_ingredients": dish.get("common_ingredients"),
            "safety_status": dish["safety_status"],
            "conflict_reason": dish.get("reason"),
            "possible_substitution": dish.get("possible_substitutions"),
        }
        for dish in state["final_validated_candidates"]
    ]

    return {
        "user_preferences": {
            "meal_time": preferences.get("meal_time"),
            "dietary_preference": preferences.get("dietary_preference"),
            "spice_preference": preferences.get("spice_preference"),
            "city": preferences.get("city"),
            "locality": preferences.get("locality"),
        },
        "dietary_restrictions": preferences.get("restrictions", []),
        "predicted_cuisine": selected_cuisine,
        "final_budget": state["final_budget"],
        "candidates": candidates,
    }


def generate_recommendation(context, client=None):
    """Ask Gemini to explain the supplied validated candidates."""
    if not context.get("candidates"):
        raise ValueError("Gemini cannot be called without valid candidates")

    if client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Set GEMINI_API_KEY before requesting a recommendation")
        client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=json.dumps(context, indent=2, ensure_ascii=False),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
            max_output_tokens=800,
        ),
    )

    final_response = (response.text or "").strip()
    if not final_response:
        raise RuntimeError("Gemini returned an empty recommendation")
    return final_response
