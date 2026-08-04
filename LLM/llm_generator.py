"""Generate a grounded recommendation from validated dish candidates."""

import json

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI


MODEL_NAME = "gemini-3.6-flash"

SYSTEM_INSTRUCTIONS = """
You explain validated culinary recommendations. Always call the
get_validated_candidates tool before answering.

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


def generate_recommendation(context):
    """Use a simple LangChain agent to explain validated candidates."""

    @tool
    def get_validated_candidates() -> str:
        """Return the user profile and dishes already validated by safety rules."""
        return json.dumps(context, indent=2, ensure_ascii=False)

    model = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        thinking_level="minimal",
        max_tokens=1200,
    )
    agent = create_agent(
        model=model,
        tools=[get_validated_candidates],
        system_prompt=SYSTEM_INSTRUCTIONS,
    )
    result = agent.invoke({
        "messages": [{
            "role": "user",
            "content": "Use the validated candidates and recommend the best meal.",
        }]
    })
    return str(result["messages"][-1].text)
