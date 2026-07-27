"""Inference helper for expected order value predictions."""

from __future__ import annotations

from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REQUIRED_FEATURES = {
    "user_average_order_value",
    "meal_time",
    "weekday_or_weekend",
    "location",
    "payment_method",
    "previous_order_count",
    "preferred_cuisine",
}


class SpendingPredictor:
    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.model = joblib.load(models_dir / "budget_regressor.pkl")
        self.preprocessor = joblib.load(models_dir / "spending_preprocessor.pkl")

    def predict_expected_order_value(self, user_context: dict) -> float:
        missing = REQUIRED_FEATURES - set(user_context)
        if missing:
            raise ValueError(f"Missing spending features: {sorted(missing)}")
        if int(user_context["previous_order_count"]) < 1:
            raise ValueError(
                "Spending prediction requires at least one previous order; "
                "use the onboarding budget for cold-start users."
            )
        prepared = {
            "user_average_order_value": float(user_context["user_average_order_value"]),
            "meal_time": str(user_context["meal_time"]),
            "weekday_or_weekend": str(user_context["weekday_or_weekend"]),
            "location": str(user_context["location"]),
            "payment_method": str(user_context["payment_method"]),
            "previous_order_count": int(user_context["previous_order_count"]),
            "preferred_cuisine": str(user_context["preferred_cuisine"]),
        }
        matrix = self.preprocessor.transform([prepared])
        prediction = max(0.0, float(self.model.predict(matrix)[0]))
        return round(prediction, 2)

    def predict_spending_limit(self, user_context: dict) -> int:
        return int(round(self.predict_expected_order_value(user_context)))
