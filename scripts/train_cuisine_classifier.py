"""Benchmark cuisine classifiers and save the validation-selected winner."""

from __future__ import annotations

import csv
import gc
import json
import os
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
JOBLIB_TEMP_DIR = PROJECT_ROOT / ".joblib_temp"
JOBLIB_TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(JOBLIB_TEMP_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from scipy.sparse import csr_matrix
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


DATA_PATH = PROJECT_ROOT / "data" / "user_order_history.csv"
TAXONOMY_PATH = PROJECT_ROOT / "data" / "cuisine_taxonomy.csv"
FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "cuisine_classification_features.csv"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_STATE = 42
MINIMUM_ACCURACY = 0.75
MINIMUM_HISTORY_ORDERS = 1

FEATURE_COLUMNS = [
    "user_id",
    "meal_time",
    "day_type",
    "vegetarian",
    "average_previous_rating",
    "most_ordered_cuisine",
    "order_frequency",
    "spice_preference",
    "previous_cuisine_orders",
]
TARGET_COLUMN = "ordered_cuisine"


def read_taxonomy(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        labels = [row["cuisine"].strip() for row in csv.DictReader(handle)]
    if not labels or len(labels) != len(set(labels)):
        raise ValueError("Cuisine taxonomy is empty or contains duplicates")
    return labels


def engineer_historical_features(data_path: Path, expected_classes: set[str]) -> tuple[list[dict], list[str]]:
    """Build each row using only information available before its current order."""
    user_state = defaultdict(
        lambda: {"orders": 0, "rating_sum": 0.0, "cuisine_counts": Counter()}
    )
    features: list[dict] = []
    targets: list[str] = []

    with data_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {
            "user_id", "meal_time", "day_type", "vegetarian", "spice_preference",
            "rating", TARGET_COLUMN,
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Order history is missing columns: {sorted(missing)}")

        for row_number, row in enumerate(reader, start=2):
            user_id = row["user_id"].strip()
            target = row[TARGET_COLUMN].strip()
            state = user_state[user_id]
            prior_counts: Counter = state["cuisine_counts"]
            if prior_counts:
                most_ordered, previous_cuisine_orders = prior_counts.most_common(1)[0]
            else:
                most_ordered, previous_cuisine_orders = "Unknown", 0
            average_rating = state["rating_sum"] / state["orders"] if state["orders"] else 4.0
            feature_row = {
                "user_id": user_id,
                "meal_time": row["meal_time"].strip(),
                "day_type": row["day_type"].strip(),
                "vegetarian": row["vegetarian"].strip(),
                "average_previous_rating": round(average_rating, 3),
                "most_ordered_cuisine": most_ordered,
                "order_frequency": state["orders"],
                "spice_preference": row["spice_preference"].strip(),
                "previous_cuisine_orders": previous_cuisine_orders,
            }
            if not all(str(value).strip() for value in feature_row.values()):
                raise ValueError(f"Blank feature generated at source row {row_number}")
            features.append(feature_row)
            targets.append(target)
            state["orders"] += 1
            state["rating_sum"] += float(row["rating"])
            prior_counts[target] += 1

    found_classes = set(targets)
    if found_classes != expected_classes:
        missing = expected_classes - found_classes
        extra = found_classes - expected_classes
        raise ValueError(f"Target/taxonomy mismatch. Missing={sorted(missing)}, extra={sorted(extra)}")
    return features, targets


def save_feature_dataset(features: list[dict], targets: list[str]) -> None:
    with FEATURE_DATA_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS + [TARGET_COLUMN])
        writer.writeheader()
        for feature_row, target in zip(features, targets):
            writer.writerow({**feature_row, TARGET_COLUMN: target})


def metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def model_suite(class_count: int) -> dict[str, object]:
    return {
        "Dummy Most Frequent": DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Ridge Classifier": RidgeClassifier(class_weight="balanced"),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
        "RBF SVM": SVC(
            C=3.0, kernel="rbf", gamma="scale", class_weight="balanced", cache_size=500
        ),
        "SGD Logistic": SGDClassifier(
            loss="log_loss", max_iter=2000, class_weight="balanced",
            early_stopping=True, random_state=RANDOM_STATE,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=1),
        "Complement Naive Bayes": ComplementNB(alpha=0.5),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=35, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=80, max_depth=25, min_samples_leaf=1, max_features=0.20,
            class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=35, min_samples_leaf=1, max_features=0.35,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE,
        ),
        "XGBoost": XGBClassifier(
            objective="multi:softprob", num_class=class_count, n_estimators=35,
            max_depth=4, max_bin=64, learning_rate=0.18, subsample=0.9,
            colsample_bytree=0.75, tree_method="hist", n_jobs=2,
            random_state=RANDOM_STATE,
        ),
    }


def fit_model(name: str, estimator, x, y):
    if name == "XGBoost":
        sample_weight = compute_sample_weight(class_weight="balanced", y=y)
        return estimator.fit(x, y, sample_weight=sample_weight)
    return estimator.fit(x, y)


def compact_sparse(matrix):
    """Return CSR data with 32-bit indices for SVM/SGD compatibility."""
    matrix = matrix.tocsr()
    return csr_matrix(
        (
            matrix.data,
            matrix.indices.astype(np.int32, copy=False),
            matrix.indptr.astype(np.int32, copy=False),
        ),
        shape=matrix.shape,
    )


def predict_in_batches(estimator, matrix, batch_size: int = 250):
    predictions = []
    for start in range(0, matrix.shape[0], batch_size):
        predictions.append(estimator.predict(matrix[start : start + batch_size]))
    return np.concatenate(predictions)


def benchmark_models(models, x_train, y_train, x_validation, y_validation) -> list[dict]:
    results = []
    for name, prototype in models.items():
        started = time.perf_counter()
        fitted = None
        try:
            estimator = clone(prototype)
            fitted = fit_model(name, estimator, x_train, y_train)
            predictions = (
                predict_in_batches(fitted, x_validation)
                if name == "K-Nearest Neighbors"
                else fitted.predict(x_validation)
            )
            result = {
                "model": name,
                "status": "completed",
                "fit_seconds": round(time.perf_counter() - started, 3),
                **metrics(y_validation, predictions),
            }
            print(
                f"{name:24s} accuracy={result['accuracy']:.4f} "
                f"weighted_f1={result['f1_weighted']:.4f} time={result['fit_seconds']:.1f}s"
            )
        except Exception as exc:
            result = {
                "model": name,
                "status": f"failed: {type(exc).__name__}: {exc}",
                "fit_seconds": round(time.perf_counter() - started, 3),
                "accuracy": 0.0,
                "precision_weighted": 0.0,
                "recall_weighted": 0.0,
                "f1_weighted": 0.0,
                "f1_macro": 0.0,
            }
            print(f"{name:24s} FAILED: {exc}")
        results.append(result)
        del fitted
        gc.collect()
    return sorted(results, key=lambda row: (row["accuracy"], row["f1_weighted"]), reverse=True)


def save_comparison(results: list[dict]) -> None:
    fields = [
        "rank", "model", "status", "accuracy", "precision_weighted",
        "recall_weighted", "f1_weighted", "f1_macro", "fit_seconds",
    ]
    with (REPORTS_DIR / "model_comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in enumerate(results, start=1):
            writer.writerow({"rank": rank, **row})
    with (REPORTS_DIR / "model_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


def save_evaluation(y_true, y_pred, class_names: list[str]) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    with (REPORTS_DIR / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual/predicted", *class_names])
        for class_name, values in zip(class_names, matrix):
            writer.writerow([class_name, *values])

    normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(16, 14))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axis.set_title("Normalized Confusion Matrix - All Cuisine Categories", fontsize=18)
    axis.set_xlabel("Predicted cuisine")
    axis.set_ylabel("Actual cuisine")
    axis.set_xticks(range(len(class_names)), class_names, rotation=90, fontsize=5)
    axis.set_yticks(range(len(class_names)), class_names, fontsize=5)
    figure.colorbar(image, ax=axis, fraction=0.025, pad=0.01)
    figure.tight_layout()
    figure.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=100)
    plt.close(figure)

    report = classification_report(
        y_true, y_pred, labels=range(len(class_names)), target_names=class_names,
        output_dict=True, zero_division=0,
    )
    with (REPORTS_DIR / "classification_report.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class", "precision", "recall", "f1_score", "support"])
        for class_name in class_names:
            values = report[class_name]
            writer.writerow([
                class_name, round(values["precision"], 6), round(values["recall"], 6),
                round(values["f1-score"], 6), int(values["support"]),
            ])


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    taxonomy = read_taxonomy(TAXONOMY_PATH)
    all_features, all_targets = engineer_historical_features(DATA_PATH, set(taxonomy))
    save_feature_dataset(all_features, all_targets)

    eligible = [
        (feature_row, target)
        for feature_row, target in zip(all_features, all_targets)
        if feature_row["order_frequency"] >= MINIMUM_HISTORY_ORDERS
    ]
    features = [item[0] for item in eligible]
    targets = [item[1] for item in eligible]
    excluded_cold_start_rows = len(all_features) - len(features)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(targets)
    x_development_rows, x_test_rows, y_development, y_test = train_test_split(
        features, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y,
    )
    x_train_rows, x_validation_rows, y_train, y_validation = train_test_split(
        x_development_rows, y_development, test_size=0.1764705882,
        random_state=RANDOM_STATE, stratify=y_development,
    )

    benchmark_preprocessor = DictVectorizer(sparse=True)
    x_train = compact_sparse(benchmark_preprocessor.fit_transform(x_train_rows))
    x_validation = compact_sparse(benchmark_preprocessor.transform(x_validation_rows))
    models = model_suite(len(label_encoder.classes_))
    results = benchmark_models(models, x_train, y_train, x_validation, y_validation)
    save_comparison(results)
    completed = [row for row in results if row["status"] == "completed"]
    if not completed:
        raise RuntimeError("Every classifier failed")
    winner_name = completed[0]["model"]
    print(f"Validation winner: {winner_name}")
    del x_train, x_validation, benchmark_preprocessor
    gc.collect()

    final_preprocessor = DictVectorizer(sparse=True)
    x_development = compact_sparse(final_preprocessor.fit_transform(x_development_rows))
    x_test = compact_sparse(final_preprocessor.transform(x_test_rows))
    selected_estimator = clone(models[winner_name])
    if hasattr(selected_estimator, "predict_proba"):
        final_model = fit_model(winner_name, selected_estimator, x_development, y_development)
    else:
        calibrated = CalibratedClassifierCV(selected_estimator, method="sigmoid", cv=3, n_jobs=1)
        final_model = calibrated.fit(x_development, y_development)

    test_predictions = final_model.predict(x_test)
    final_metrics = metrics(y_test, test_predictions)
    if final_metrics["accuracy"] < MINIMUM_ACCURACY:
        raise RuntimeError(
            f"Winning model test accuracy {final_metrics['accuracy']:.3f} is below {MINIMUM_ACCURACY:.2f}"
        )

    joblib.dump(final_model, MODELS_DIR / "cuisine_classifier.pkl")
    joblib.dump(final_preprocessor, MODELS_DIR / "preprocessors.pkl")
    joblib.dump(label_encoder, MODELS_DIR / "label_encoder.pkl")
    obsolete_baseline = MODELS_DIR / "logistic_baseline.pkl"
    if obsolete_baseline.exists():
        obsolete_baseline.unlink()

    metadata = {
        "selected_model": winner_name,
        "selection_method": "highest validation accuracy, weighted F1 as tie-breaker",
        "random_state": RANDOM_STATE,
        "minimum_required_accuracy": MINIMUM_ACCURACY,
        "minimum_history_orders": MINIMUM_HISTORY_ORDERS,
        "excluded_cold_start_rows": excluded_cold_start_rows,
        "training_rows": len(x_train_rows),
        "validation_rows": len(x_validation_rows),
        "test_rows": len(x_test_rows),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "class_count": len(label_encoder.classes_),
        "classes": label_encoder.classes_.tolist(),
        "scikit_learn_version": sklearn.__version__,
        "benchmark_results": results,
        "final_test_metrics": final_metrics,
    }
    with (MODELS_DIR / "model_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with (REPORTS_DIR / "classification_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    del x_development, x_development_rows, y_development, results
    gc.collect()
    save_evaluation(y_test, test_predictions, label_encoder.classes_.tolist())

    print(f"Final test accuracy: {final_metrics['accuracy']:.4f}")
    print(f"Final weighted F1: {final_metrics['f1_weighted']:.4f}")
    print(f"Cuisine classes: {len(label_encoder.classes_)}")


if __name__ == "__main__":
    main()
