"""Evaluate exact matching, semantic ranking, and unknown-query rejection."""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rag.rag_pipeline import CulinaryRAG


SEMANTIC_CASES = [
    ("Shahi Butter Paneer Curry", {"Shahi Paneer", "Paneer Butter Masala"}),
    ("Chickpea Masala", {"Chana Masala", "Chole Masala"}),
    ("Cottage Cheese Spinach Curry", {"Paneer Palak", "Palak Paneer"}),
    ("Lentil Tadka", {"Dal Tadka", "Dal Fry Tadka", "Yellow Dal Tadka"}),
    ("Kidney Bean Masala", {"Rajma Masala", "Rajma"}),
    (
        "Boneless Chicken Butter Masala",
        {"Boneless Chicken Masala", "Chicken Butter Masala", "Butter Chicken Boneless"},
    ),
    ("Vegetable Hakka Noodle", {"Hakka Noodles", "Veg Hakka Noodles"}),
    ("Steamed Rice and Lentil Cake", {"Idli", "Plain Idli", "Rice Idli"}),
    ("Flattened Rice Breakfast Dish", {"Poha", "Plain Poha", "Kanda Poha"}),
    ("Semolina Breakfast Dish", {"Upma", "Veg Upma", "Vegetable Upma"}),
    ("Grilled Cottage Cheese Cubes", {"Paneer Tikka", "Paneer Tikka Dry"}),
    ("Yogurt Cucumber Side Dish", {"Cucumber Raita", "Raita"}),
    ("Spiced Potato Cauliflower Curry", {"Aloo Gobi", "Aloo Gobi Masala"}),
    ("Dum Cooked Vegetable Rice", {"Veg Dum Biryani", "Veg Biryani", "Dum Biryani"}),
    (
        "Crispy Chickpea Flour Fritters",
        {"Pakoda", "Pakora", "Mix Pakoda", "Mix Pakora"},
    ),
]

UNKNOWN_QUERIES = [
    "Royal Mystery Bowl",
    "Chocolate Astronaut Sushi",
    "Blueberry Breakfast Cereal",
    "Grilled Salmon Avocado Tacos",
    "Peanut Butter Jelly Sandwich",
    "Beef Cheeseburger",
    "Tuna Poke Bowl",
    "Maple Bacon Pancakes",
    "Lobster Thermidor",
    "Quinoa Avocado Smoothie",
]


def evaluate(rag: CulinaryRAG) -> dict:
    exact_rows = rag.frame.head(100).to_dict(orient="records")
    exact_hits = sum(
        rag.retrieve(f"Special {row['dish_name']} Full 500 ml").matched_dish_id
        == str(row["dish_id"])
        for row in exact_rows
    )

    ranks = []
    accepted_correct = 0
    accepted_count = 0
    semantic_details = []
    for query, acceptable in SEMANTIC_CASES:
        candidates = rag.semantic_candidates(query, top_k=3)
        names = [candidate["dish_name"] for candidate in candidates]
        rank = next(
            (index + 1 for index, name in enumerate(names) if name in acceptable),
            None,
        )
        result = rag.retrieve(query)
        ranks.append(rank)
        if result.retrieval_status == "matched":
            accepted_count += 1
            accepted_correct += result.matched_dish in acceptable
        semantic_details.append(
            {
                "query": query,
                "top_3": names,
                "relevant_rank": rank,
                "top_distance": candidates[0]["distance"],
                "retrieval_status": result.retrieval_status,
                "accepted_dish": result.matched_dish,
            }
        )

    unknown_results = [rag.retrieve(query) for query in UNKNOWN_QUERIES]
    metrics = {
        "exact_accuracy": exact_hits / len(exact_rows),
        "semantic_precision_at_1": sum(rank == 1 for rank in ranks) / len(ranks),
        "semantic_recall_at_3": sum(rank is not None for rank in ranks) / len(ranks),
        "semantic_mrr": sum(1 / rank if rank else 0 for rank in ranks) / len(ranks),
        "accepted_match_accuracy": (
            accepted_correct / accepted_count if accepted_count else 0.0
        ),
        "accepted_semantic_queries": accepted_count,
        "semantic_query_count": len(SEMANTIC_CASES),
        "unknown_false_acceptance_rate": sum(
            result.retrieval_status == "matched" for result in unknown_results
        )
        / len(unknown_results),
        "distance_threshold": rag.distance_threshold,
    }
    return {
        "metrics": metrics,
        "semantic_cases": semantic_details,
        "unknown_cases": [result.to_dict() for result in unknown_results],
    }


def main() -> None:
    evaluation = evaluate(CulinaryRAG())
    output_path = Path(__file__).with_name("retrieval_evaluation.json")
    output_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    metrics = evaluation["metrics"]
    assert metrics["exact_accuracy"] == 1.0
    assert metrics["accepted_match_accuracy"] == 1.0
    assert metrics["unknown_false_acceptance_rate"] == 0.0
    print(json.dumps(metrics, indent=2))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
