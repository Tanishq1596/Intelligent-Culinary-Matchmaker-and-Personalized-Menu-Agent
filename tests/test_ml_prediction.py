"""Smoke test for loading the saved cuisine model and producing probabilities."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_prediction import CuisinePredictor


def main() -> None:
    predictor = CuisinePredictor()
    result = predictor.predict_probabilities(
        {
            "user_id": "U0042",
            "meal_time": "Dinner",
            "day_type": "Weekend",
            "vegetarian": "Yes",
            "average_previous_rating": 4.4,
            "most_ordered_cuisine": "North Indian",
            "order_frequency": 12,
            "spice_preference": "Hot",
            "previous_cuisine_orders": 9,
        }
    )
    assert len(result) == 82
    assert abs(sum(item["probability"] for item in result) - 1.0) < 0.01
    assert result == sorted(result, key=lambda item: item["probability"], reverse=True)
    print(result)


if __name__ == "__main__":
    main()
