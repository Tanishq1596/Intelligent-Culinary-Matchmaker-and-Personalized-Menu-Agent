"""Inference helper for the trained cuisine classifier."""

from __future__ import annotations

from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
MINIMUM_HISTORY_ORDERS = 3
REQUIRED_FEATURES = {
    "meal_time",
    "day_type",
    "vegetarian",
    "average_previous_rating",
    "most_ordered_cuisine",
    "order_frequency",
    "spice_preference",
    "previous_cuisine_orders",
}


class CuisinePredictor:
    def __init__(self, models_dir: Path = MODELS_DIR) -> None:
        self.model = joblib.load(models_dir / "cuisine_classifier.pkl")
        self.preprocessor = joblib.load(models_dir / "preprocessors.pkl")
        self.label_encoder = joblib.load(models_dir / "label_encoder.pkl")

    def predict_probabilities(self, user_features: dict) -> list[dict[str, float | str]]:
        missing = REQUIRED_FEATURES - set(user_features)
        if missing:
            raise ValueError(f"Missing classifier features: {sorted(missing)}")
        if int(user_features["order_frequency"]) < MINIMUM_HISTORY_ORDERS:
            raise ValueError(
                f"Cuisine classification requires at least {MINIMUM_HISTORY_ORDERS} "
                "previous orders; "
                "use onboarding preferences for cold-start users."
            )

        prepared = {
            "meal_time": str(user_features["meal_time"]),
            "day_type": str(user_features["day_type"]),
            "vegetarian": str(user_features["vegetarian"]),
            "average_previous_rating": float(user_features["average_previous_rating"]),
            "most_ordered_cuisine": str(user_features["most_ordered_cuisine"]),
            "order_frequency": int(user_features["order_frequency"]),
            "spice_preference": str(user_features["spice_preference"]),
            "previous_cuisine_orders": int(user_features["previous_cuisine_orders"]),
        }
        matrix = self.preprocessor.transform([prepared])
        probabilities = self.model.predict_proba(matrix)[0]
        class_names = self.label_encoder.inverse_transform(self.model.classes_)
        ranked = sorted(
            (
                {"cuisine": str(cuisine), "probability": round(float(probability), 4)}
                for cuisine, probability in zip(class_names, probabilities)
            ),
            key=lambda item: item["probability"],
            reverse=True,
        )
        return ranked

    def predict(self, user_features: dict) -> str:
        return str(self.predict_probabilities(user_features)[0]["cuisine"])
