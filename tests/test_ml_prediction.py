"""Smoke test for loading the saved cuisine model and producing probabilities."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ml_prediction import CuisinePredictor


def main() -> None:
    predictor = CuisinePredictor()
    user_features = {
        "meal_time": "Dinner",
        "day_type": "Weekend",
        "vegetarian": "Yes",
        "average_previous_rating": 4.4,
        "most_ordered_cuisine": "North Indian",
        "order_frequency": 12,
        "spice_preference": "Hot",
        "previous_cuisine_orders": 9,
    }
    result = predictor.predict_probabilities(user_features)
    assert len(result) == 82
    assert abs(sum(item["probability"] for item in result) - 1.0) < 0.01
    assert result == sorted(result, key=lambda item: item["probability"], reverse=True)
    assert not any(
        feature.startswith("user_id=")
        for feature in predictor.preprocessor.get_feature_names_out()
    )

    try:
        predictor.predict_probabilities({**user_features, "order_frequency": 2})
    except ValueError as error:
        assert "onboarding preferences" in str(error)
    else:
        raise AssertionError("Users below the history threshold must use onboarding")
    print(result)


if __name__ == "__main__":
    main()
