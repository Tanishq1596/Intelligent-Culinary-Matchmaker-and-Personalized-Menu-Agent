# Culinary Agent Workflow

This folder contains the structured workflow that connects the trained models,
restaurant filtering, RAG retrieval, and deterministic safety rules.

Gemini prompt construction and response generation are kept separately in
`../LLM/llm_generator.py`.

## Why StateGraph

The workflow uses LangGraph `StateGraph` because routing must remain predictable:

1. Parse the request.
2. Predict the top cuisines.
3. Use an explicit budget or call the spending model.
4. Filter real restaurant dishes.
5. Try the second cuisine once when the first has no candidates.
6. Retrieve culinary knowledge.
7. Apply deterministic safety rules.
8. Exclude rejected and unreliable dishes.
9. End with a fixed no-match response when no valid candidates remain.
10. Rank compatible dishes before warning candidates.
11. Send only the structured validated candidates to Gemini 3.6 Flash.
12. Store Gemini's explanation in `final_response` for Streamlit.

`create_agent` is intentionally not used here. It creates an LLM-driven
tool loop, while this workflow requires fixed tool order and safety decisions that
an LLM cannot override. Gemini only explains the already validated result.

## Gemini setup

Set your Gemini API key before running the workflow:

```powershell
$env:GEMINI_API_KEY="your-api-key"
```

The generator uses the stable `gemini-3.6-flash` model. Sampling parameters are
not supplied because Gemini 3.6 deprecates `temperature`, `top_p`, and `top_k`.
Grounding is controlled through strict instructions and filtered context.

## Run from Python

Add `culinary_matchmaker` to the Python path and provide known history features:

```python
from agentic_ai import run_workflow

profile = {
    "classifier_features": {
        "user_id": "U00042",
        "meal_time": "Dinner",
        "day_type": "Weekend",
        "vegetarian": "Yes",
        "average_previous_rating": 4.4,
        "most_ordered_cuisine": "North Indian",
        "order_frequency": 12,
        "spice_preference": "Hot",
        "previous_cuisine_orders": 9,
    },
    "spending_features": {
        "user_average_order_value": 240,
        "meal_time": "Dinner",
        "weekday_or_weekend": "Weekend",
        "location": "Bengaluru",
        "payment_method": "UPI",
        "previous_order_count": 12,
        "preferred_cuisine": "North Indian",
    },
}

result = run_workflow(
    "Suggest a spicy vegetarian dinner in HSR, Bengaluru under ₹250. "
    "I am lactose intolerant and allergic to peanuts.",
    profile,
)
```

The result is a dictionary matching `WorkflowState`. Streamlit should render
`final_response`; it contains either Gemini's grounded explanation or the fixed
no-match message. Gemini is never called when no validated candidate exists.
