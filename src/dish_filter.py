"""Reusable Pandas filtering for real restaurant-dish candidates."""

from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_LOCATIONS = [
    PROJECT_ROOT / "data" / "swiggy_cleaned_sample_expanded.csv",
    PROJECT_ROOT / "data" / "swiggy_cleaned.csv",
    PROJECT_ROOT.parent / "swiggy_cleaned_sample_expanded.csv",
]
REQUIRED_COLUMNS = [
    "restaurant_id", "restaurant_name", "city", "locality", "address",
    "cuisine", "menu_category", "dish_name", "dish_price", "veg_nonveg",
    "restaurant_rating", "rating_count",
]


def text_key(value):
    return " ".join(str(value).strip().casefold().split())


def search_key(value):
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(value).casefold()).split())


def cuisine_key(value):
    labels = {text_key(label) for label in str(value).split(",")}
    labels.discard("")
    return "|" + "|".join(sorted(labels)) + "|"


def dietary_key(preference):
    if preference is None or text_key(preference) in {"", "any", "no preference"}:
        return None
    key = text_key(preference).replace("_", "-")
    aliases = {
        "veg": "veg", "vegetarian": "veg",
        "non-veg": "non-veg", "non veg": "non-veg",
        "nonvegetarian": "non-veg", "non-vegetarian": "non-veg",
    }
    if key not in aliases:
        raise ValueError("dietary_preference must be Veg, Non-veg, or None")
    return aliases[key]


class DishCatalog:
    """Load the cleaned Swiggy data once and return grounded candidates."""

    def __init__(self, data_path=None):
        self.data_path = Path(data_path) if data_path else next(
            (path for path in DATA_LOCATIONS if path.is_file()), None
        )
        if self.data_path is None:
            raise FileNotFoundError(f"Swiggy dataset not found in: {DATA_LOCATIONS}")

        dishes = pd.read_csv(self.data_path, usecols=REQUIRED_COLUMNS, low_memory=False)
        self.dishes = self._prepare(dishes)
        self.city_names = {
            search_key(city): city for city in self.dishes["city"].dropna().unique()
        }
        self.locality_names = {
            search_key(locality): locality
            for locality in self.dishes["locality"].dropna().unique()
        }
        self.locality_cities = {}
        for locality, city in self.dishes[["locality", "city"]].dropna().itertuples(index=False):
            self.locality_cities.setdefault(search_key(locality), set()).add(str(city))

    @staticmethod
    def _prepare(frame):
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Restaurant dataset is missing columns: {sorted(missing)}")

        cleaned = frame[REQUIRED_COLUMNS].copy()
        cleaned["dish_price"] = pd.to_numeric(cleaned["dish_price"], errors="coerce")
        cleaned["restaurant_rating"] = pd.to_numeric(
            cleaned["restaurant_rating"], errors="coerce"
        )
        cleaned["rating_count"] = pd.to_numeric(
            cleaned["rating_count"], errors="coerce"
        ).fillna(0)
        cleaned = cleaned.dropna(
            subset=["restaurant_id", "restaurant_name", "city", "dish_name", "dish_price"]
        )
        cleaned = cleaned[cleaned["dish_price"] >= 0].copy()
        cleaned["_city_key"] = cleaned["city"].map(text_key)
        cleaned["_locality_key"] = cleaned["locality"].map(text_key)
        cleaned["_dietary_key"] = cleaned["veg_nonveg"].map(text_key)
        cleaned["_cuisine_key"] = cleaned["cuisine"].map(cuisine_key)
        cleaned["_dish_key"] = cleaned["dish_name"].map(text_key)
        cleaned = cleaned.drop_duplicates(["restaurant_id", "_dish_key"])
        return cleaned.reset_index(drop=True)

    def find_location(self, request):
        """Find a known city/locality mentioned in a natural-language request."""
        request_key = f" {search_key(request)} "
        city = self._longest_match(request_key, self.city_names)
        locality = self._longest_match(request_key, self.locality_names)

        if locality and not city:
            possible_cities = self.locality_cities.get(search_key(locality), set())
            if len(possible_cities) == 1:
                city = next(iter(possible_cities))
        return city, locality

    @staticmethod
    def _longest_match(request_key, names):
        matches = [actual for key, actual in names.items() if f" {key} " in request_key]
        return max(matches, key=lambda value: len(str(value)), default=None)

    def filter(self, *, city, locality, cuisine, budget, dietary_preference=None,
               minimum_rating=None, top_n=10):
        """Apply hard constraints and rank the best matching real dishes."""
        if not city or not cuisine:
            raise ValueError("city and cuisine are required")
        budget = float(budget)
        if budget <= 0:
            raise ValueError("budget must be positive")
        if not 1 <= int(top_n) <= 100:
            raise ValueError("top_n must be between 1 and 100")

        candidates = self.dishes[
            (self.dishes["_city_key"] == text_key(city))
            & (self.dishes["dish_price"] <= budget)
            & self.dishes["_cuisine_key"].str.contains(
                f"|{text_key(cuisine)}|", regex=False
            )
        ]
        if locality:
            candidates = candidates[candidates["_locality_key"] == text_key(locality)]

        preference = dietary_key(dietary_preference)
        if preference:
            candidates = candidates[candidates["_dietary_key"] == preference]
        if minimum_rating is not None:
            rating = float(minimum_rating)
            if not 0 <= rating <= 5:
                raise ValueError("minimum_rating must be between 0 and 5")
            candidates = candidates[candidates["restaurant_rating"] >= rating]

        candidates = candidates.sort_values(
            ["restaurant_rating", "rating_count", "dish_price"],
            ascending=[False, False, True],
            na_position="last",
            kind="stable",
        ).head(int(top_n))
        result = candidates[REQUIRED_COLUMNS].reset_index(drop=True).copy()
        result.insert(0, "rank", range(1, len(result) + 1))
        result = result.astype(object).where(pd.notna(result), None)
        return result.to_dict(orient="records")
