"""Build an all-category cuisine taxonomy and synthetic user-order history."""

from __future__ import annotations

import csv
import math
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SWIGGY_PATH = REPOSITORY_ROOT / "swiggy_cleaned_sample_expanded.csv"
TAXONOMY_PATH = PROJECT_ROOT / "data" / "cuisine_taxonomy.csv"
HISTORY_PATH = PROJECT_ROOT / "data" / "user_order_history.csv"
ROOT_HISTORY_PATH = REPOSITORY_ROOT / "synthetic_user_order_history.csv"
RANDOM_STATE = 42
USER_COUNT = 5000
ORDER_COUNT = 40000
MINIMUM_PROFILES_PER_CUISINE = 20
CITIES = [
    "Bengaluru", "Pune", "Mumbai", "Delhi", "Hyderabad",
    "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow",
]
LOCATION_FACTORS = {
    "Bengaluru": 1.10, "Pune": 1.03, "Mumbai": 1.15, "Delhi": 1.11,
    "Hyderabad": 1.04, "Chennai": 1.02, "Kolkata": 0.96,
    "Ahmedabad": 0.98, "Jaipur": 0.94, "Lucknow": 0.92,
}
PAYMENT_SEQUENCE = [
    "UPI", "UPI", "UPI", "UPI", "UPI",
    "Cash on Delivery", "Cash on Delivery", "Credit Card", "Credit Card", "Wallet",
]
PAYMENT_FACTORS = {
    "UPI": 1.00,
    "Cash on Delivery": 0.96,
    "Credit Card": 1.05,
    "Wallet": 0.98,
}


def split_cuisines(value: str) -> list[str]:
    labels = []
    for part in re.split(r"[,/|]+", value or ""):
        label = re.sub(r"\s+", " ", part).strip()
        if label:
            labels.append(label)
    return labels


def build_taxonomy(source_path: Path) -> list[dict]:
    counts: Counter[str] = Counter()
    prices: dict[str, list[int]] = defaultdict(list)
    total_rows = 0

    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"cuisine", "dish_price"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Swiggy data is missing columns: {sorted(missing)}")

        for row in reader:
            total_rows += 1
            try:
                dish_price = int(row["dish_price"])
            except ValueError as exc:
                raise ValueError(f"Non-integer dish price found: {row['dish_price']!r}") from exc
            for cuisine in split_cuisines(row["cuisine"]):
                counts[cuisine] += 1
                prices[cuisine].append(dish_price)

    taxonomy = [
        {
            "cuisine": cuisine,
            "source_record_count": counts[cuisine],
            "source_share": round(counts[cuisine] / total_rows, 8),
            "median_dish_price": int(round(median(prices[cuisine]))),
        }
        for cuisine in sorted(counts)
    ]
    if not taxonomy:
        raise ValueError("No cuisine labels were found")
    return taxonomy


def write_taxonomy(taxonomy: list[dict], output_path: Path) -> None:
    fields = ["cuisine", "source_record_count", "source_share", "median_dish_price"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(taxonomy)


def choose_spice_preference(rng: random.Random, cuisine: str) -> str:
    mild_categories = {
        "Bakery", "Beverages", "Desserts", "Ice Cream", "Ice Cream Cakes",
        "Juices", "Paan", "Sweets", "Waffle",
    }
    hot_categories = {
        "Andhra", "Biryani", "Chettinad", "Hyderabadi", "Kebabs", "Lucknowi",
        "Mexican", "Mughlai", "Naga", "North Eastern", "Punjabi", "Tandoor",
    }
    if cuisine in mild_categories:
        weights = [0.88, 0.10, 0.02]
    elif cuisine in hot_categories:
        weights = [0.08, 0.40, 0.52]
    else:
        weights = [0.20, 0.55, 0.25]
    return rng.choices(["Mild", "Medium", "Hot"], weights=weights, k=1)[0]


def build_profiles(rng: random.Random, taxonomy: list[dict]) -> list[dict]:
    cuisines = [row["cuisine"] for row in taxonomy]
    source_weights = [math.sqrt(row["source_record_count"]) for row in taxonomy]
    profile_cuisines = [
        cuisine
        for cuisine in cuisines
        for _ in range(MINIMUM_PROFILES_PER_CUISINE)
    ]
    remaining = USER_COUNT - len(profile_cuisines)
    if remaining < 0:
        raise ValueError("USER_COUNT is too small for the minimum profiles per cuisine")
    profile_cuisines.extend(rng.choices(cuisines, weights=source_weights, k=remaining))
    rng.shuffle(profile_cuisines)

    median_prices = {row["cuisine"]: row["median_dish_price"] for row in taxonomy}
    profiles = []
    for index, preferred_cuisine in enumerate(profile_cuisines, start=1):
        spend_center = max(80, median_prices[preferred_cuisine])
        average_spend = int(round(max(60, rng.gauss(spend_center, max(20, spend_center * 0.18))) / 5) * 5)
        profiles.append(
            {
                "user_id": f"U{index:05d}",
                "location": CITIES[(index - 1) % len(CITIES)],
                "vegetarian": "Yes" if rng.random() < 0.43 else "No",
                "preferred_cuisine": preferred_cuisine,
                "average_spend": average_spend,
                "spice_preference": choose_spice_preference(rng, preferred_cuisine),
                "loyalty": rng.uniform(0.86, 0.97),
            }
        )
    return profiles


def generate_history(taxonomy: list[dict], output_path: Path) -> dict:
    rng = random.Random(RANDOM_STATE)
    profiles = build_profiles(rng, taxonomy)
    cuisines = [row["cuisine"] for row in taxonomy]
    cuisine_weights = [math.sqrt(row["source_record_count"]) for row in taxonomy]
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    day_weights = [12, 12, 12, 13, 15, 19, 17]
    meal_times = ["Breakfast", "Lunch", "Evening Snack", "Dinner"]
    meal_weights = [14, 29, 16, 41]
    meal_factor = {"Breakfast": 0.72, "Lunch": 0.95, "Evening Snack": 0.66, "Dinner": 1.18}
    fields = [
        "order_id", "user_id", "meal_time", "day_of_week", "day_type", "vegetarian",
        "location", "payment_method", "preferred_cuisine", "average_spend",
        "spice_preference", "rating",
        "ordered_cuisine", "order_amount",
    ]
    target_counts: Counter[str] = Counter()

    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row_index in range(ORDER_COUNT):
            profile = profiles[row_index] if row_index < len(profiles) else rng.choice(profiles)
            day = rng.choices(days, weights=day_weights, k=1)[0]
            meal_time = rng.choices(meal_times, weights=meal_weights, k=1)[0]

            if rng.random() < profile["loyalty"]:
                ordered_cuisine = profile["preferred_cuisine"]
            else:
                while True:
                    ordered_cuisine = rng.choices(cuisines, weights=cuisine_weights, k=1)[0]
                    if ordered_cuisine != profile["preferred_cuisine"]:
                        break

            weekend_factor = 1.08 if day in {"Saturday", "Sunday"} else 1.0
            user_number = int(profile["user_id"][1:])
            payment_method = PAYMENT_SEQUENCE[(row_index + user_number) % len(PAYMENT_SEQUENCE)]
            amount = (
                profile["average_spend"]
                * meal_factor[meal_time]
                * weekend_factor
                * LOCATION_FACTORS[profile["location"]]
                * PAYMENT_FACTORS[payment_method]
            )
            amount = int(round(max(40, amount * rng.gauss(1.0, 0.13)) / 5) * 5)
            rating_center = 4.4 if ordered_cuisine == profile["preferred_cuisine"] else 3.6
            rating = round(min(5.0, max(1.0, rng.gauss(rating_center, 0.42))), 1)
            target_counts[ordered_cuisine] += 1
            writer.writerow(
                {
                    "order_id": f"O{row_index + 1:06d}",
                    "user_id": profile["user_id"],
                    "meal_time": meal_time,
                    "day_of_week": day,
                    "day_type": "Weekend" if day in {"Saturday", "Sunday"} else "Weekday",
                    "vegetarian": profile["vegetarian"],
                    "location": profile["location"],
                    "payment_method": payment_method,
                    "preferred_cuisine": profile["preferred_cuisine"],
                    "average_spend": profile["average_spend"],
                    "spice_preference": profile["spice_preference"],
                    "rating": rating,
                    "ordered_cuisine": ordered_cuisine,
                    "order_amount": amount,
                }
            )

    missing_targets = set(cuisines) - set(target_counts)
    if missing_targets:
        raise ValueError(f"Generated history is missing cuisine targets: {sorted(missing_targets)}")
    return {
        "rows": ORDER_COUNT,
        "users": len(profiles),
        "classes": len(target_counts),
        "minimum_class_rows": min(target_counts.values()),
        "maximum_class_rows": max(target_counts.values()),
    }


def main() -> None:
    taxonomy = build_taxonomy(SWIGGY_PATH)
    write_taxonomy(taxonomy, TAXONOMY_PATH)
    stats = generate_history(taxonomy, HISTORY_PATH)
    shutil.copyfile(HISTORY_PATH, ROOT_HISTORY_PATH)
    print(f"Cuisine taxonomy: {len(taxonomy)} classes")
    print(f"History rows: {stats['rows']}")
    print(f"Users: {stats['users']}")
    print(f"Minimum rows in one target class: {stats['minimum_class_rows']}")
    print(f"Maximum rows in one target class: {stats['maximum_class_rows']}")


if __name__ == "__main__":
    main()
