"""End-to-end smoke test for the simple LangChain workflow."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from agentic_ai import run_workflow
from agentic_ai.workflow import NO_MATCH_MESSAGE


PROFILE = {
    "classifier_features": {
        "meal_time": "Dinner",
        "day_type": "Weekend",
        "vegetarian": "Yes",
        "average_previous_rating": 4.4,
        "most_ordered_cuisine": "North Indian",
        "order_frequency": 12,
        "spice_preference": "Hot",
        "previous_cuisine_orders": 9,
    },
}


def main():
    llm_contexts = []

    def fake_llm(context):
        llm_contexts.append(context)
        return "Grounded recommendation from validated candidates."

    result = run_workflow(
        "Suggest a spicy vegetarian dinner in HSR, Bengaluru under ₹250. "
        "I am lactose intolerant and allergic to peanuts.",
        PROFILE,
        llm_generator=fake_llm,
    )

    assert result["final_budget"] == 250
    assert result["predicted_cuisine"]["primary"]["cuisine"] == "North Indian"
    assert result["predicted_cuisine"]["source"] == "history_model"
    assert len(result["filtered_candidates"]) <= 10
    assert result["rag_results"]
    assert result["safety_results"]
    assert result["final_validated_candidates"]
    assert result["final_response"] == "Grounded recommendation from validated candidates."
    assert len(llm_contexts) == 1
    assert llm_contexts[0] == result["llm_context"]
    assert all(
        item["safety_status"] in {"Likely compatible", "Possible dietary conflict"}
        for item in result["final_validated_candidates"]
    )
    assert all(
        item["price"] <= result["final_budget"]
        for item in result["final_validated_candidates"]
    )
    assert all(
        item["safety_status"] in {"Likely compatible", "Possible dietary conflict"}
        for item in result["llm_context"]["candidates"]
    )

    no_match = run_workflow(
        "Suggest a vegetarian dinner in HSR, Bengaluru under ₹1.",
        {"preferred_cuisines": ["North Indian", "Chinese"]},
        llm_generator=fake_llm,
    )
    assert not no_match["filtered_candidates"]
    assert not no_match["rag_results"]
    assert no_match["error_message"] == NO_MATCH_MESSAGE
    assert no_match["final_response"] == NO_MATCH_MESSAGE
    assert no_match["predicted_cuisine"]["source"] == "onboarding"
    assert len(llm_contexts) == 1

    print({
        "predicted_cuisine": result["predicted_cuisine"]["selected"],
        "final_budget": result["final_budget"],
        "filtered_dishes": len(result["filtered_candidates"]),
        "validated_dishes": len(result["final_validated_candidates"]),
        "llm_candidates": len(result["llm_context"]["candidates"]),
        "no_match_message": no_match["error_message"],
    })


if __name__ == "__main__":
    main()
