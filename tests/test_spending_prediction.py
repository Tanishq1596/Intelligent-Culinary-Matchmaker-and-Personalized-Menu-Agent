"""Smoke test for the saved spending regression model."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.spending_prediction import SpendingPredictor


def main() -> None:
    predictor = SpendingPredictor()
    context = {
        "user_average_order_value": 240,
        "meal_time": "Dinner",
        "weekday_or_weekend": "Weekend",
        "location": "Bengaluru",
        "payment_method": "UPI",
        "previous_order_count": 12,
        "preferred_cuisine": "North Indian",
    }
    value = predictor.predict_expected_order_value(context)
    limit = predictor.predict_spending_limit(context)
    assert value > 0
    assert limit == round(value)
    print({"expected_order_value": value, "predicted_spending_limit": limit})


if __name__ == "__main__":
    main()
