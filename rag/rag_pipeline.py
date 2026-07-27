"""Exact-first culinary RAG with a persistent ChromaDB fallback."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence
import unicodedata

import pandas as pd


# -----------------------------------------------------------------------------
# 1. Configuration
# -----------------------------------------------------------------------------

RAG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = RAG_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "culinary_knowledge.csv"
VECTOR_DB_DIR = PROJECT_ROOT / "vector_db"
MANIFEST_PATH = VECTOR_DB_DIR / "rag_manifest.json"
MODEL_CACHE_DIR = PROJECT_ROOT / ".model_cache"

COLLECTION_NAME = "culinary_dishes_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DISTANCE_THRESHOLD = 0.35
LOW_CONFIDENCE_MARGIN = 0.15
QUERY_SCHEMA_VERSION = 2

REQUIRED_COLUMNS = {
    "dish_id", "dish_name", "cuisine", "common_ingredients", "allergens",
    "dietary_tags", "description", "source",
}
OPTIONAL_COLUMNS = {
    "preparation_method", "ingredient_variations", "possible_substitutions",
    "allergen_confidence",
}

UNAVAILABLE_INGREDIENTS = "Ingredient information unavailable"
UNAVAILABLE_ALLERGENS = "Allergen information unavailable"
UNAVAILABLE_CUISINE = "Unknown cuisine"
UNAVAILABLE_SOURCE = "Unknown source"


# -----------------------------------------------------------------------------
# 2. Dish-name normalization and query preparation
# -----------------------------------------------------------------------------

BRACKETED_SERVING = re.compile(
    r"[\(\[\{][^\)\]\}]*?(?:\d|full|half|serves?|pcs?|pieces?|ml|gm|kg)"
    r"[^\)\]\}]*[\)\]\}]",
    re.IGNORECASE,
)
QUANTITY = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:ml|l|ltr|litre|g|gm|kg|pc|pcs|piece|pieces|serves?)\b",
    re.IGNORECASE,
)
MENU_MODIFIERS = re.compile(r"\b(?:special|full|half|combo)\b", re.IGNORECASE)

# These mappings connect common English food terms to names used in Indian menus.
CULINARY_SYNONYMS = [
    (("grilled cottage cheese",), "paneer tikka"),
    (("potato cauliflower", "potato and cauliflower"), "aloo gobi"),
    (("chickpea flour fritter", "chickpea flour fritters"), "pakoda pakora"),
    (("dum cooked vegetable rice", "dum cooked vegetables and rice"), "veg dum biryani"),
    (("chickpea", "chickpeas"), "chana chole"),
    (("cottage cheese",), "paneer"),
    (("spinach",), "palak saag"),
    (("lentil", "lentils"), "dal"),
    (("kidney bean", "kidney beans"), "rajma"),
    (("eggplant", "aubergine"), "baingan brinjal"),
    (("okra", "lady finger", "ladyfinger"), "bhindi"),
    (("yogurt",), "curd dahi"),
    (("flatbread",), "roti chapati"),
    (("rice lentil pancake", "rice and lentil pancake"), "dosa"),
    (("steamed rice lentil cake", "steamed rice and lentil cake"), "idli"),
    (("flattened rice",), "poha"),
    (("semolina breakfast",), "upma"),
]


def clean_text(value: Any, fallback: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def normalize_dish_name(value: Any) -> str:
    """Turn menu variants into one exact-match key."""

    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = BRACKETED_SERVING.sub(" ", text)
    text = QUANTITY.sub(" ", text)
    text = MENU_MODIFIERS.sub(" ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def semantic_query_text(query: str) -> str:
    """Format and expand a query so it resembles an indexed culinary document."""

    normalized = normalize_dish_name(query)
    synonyms = [
        replacement
        for phrases, replacement in CULINARY_SYNONYMS
        if any(phrase in normalized for phrase in phrases)
    ]
    alternatives = ", ".join(synonyms) if synonyms else "none"
    return (
        f"Dish: {query}\n"
        f"Indian dish-name equivalents: {alternatives}\n"
        f"Alternative dish terms: {alternatives}\n"
        f"Canonical Indian dish terms: {alternatives}\n"
        f"Description: {query}\n"
        f"Common ingredients or style: {query}"
    )


def cuisine_filter(cuisine: str | None) -> dict[str, Any] | None:
    """Convert one or more CSV cuisine labels into a Chroma metadata filter."""

    values = [part.strip() for part in re.split(r"[,/]", cuisine or "") if part.strip()]
    if not values:
        return None
    return {"cuisine": values[0]} if len(values) == 1 else {"cuisine": {"$in": values}}


# -----------------------------------------------------------------------------
# 3. Dataset validation, canonicalization, and document creation
# -----------------------------------------------------------------------------

def dataset_sha256(path: Path = DATA_PATH) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quality_score(row: pd.Series) -> tuple[int, int, int]:
    """Rank duplicate rows by completeness, then by restaurant support."""

    score = 0
    score += 3 if "unavailable" not in row["common_ingredients"].casefold() else 0
    score += 3 if row["allergens"] != "none_identified_from_name" else 1
    score += 2 if row["source"] else 0
    score += min(len(row["description"]) // 80, 2)
    optional_count = sum(bool(row.get(column, "")) for column in OPTIONAL_COLUMNS)
    source_count = int(row.get("source_restaurant_count", 0) or 0)
    return score, optional_count, source_count


def load_culinary_dataset(path: Path = DATA_PATH) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Validate the CSV and return one canonical row per normalized dish name."""

    dishes = pd.read_csv(path, keep_default_na=False, low_memory=False)
    missing = REQUIRED_COLUMNS - set(dishes.columns)
    if missing:
        raise ValueError(f"Culinary dataset is missing columns: {sorted(missing)}")

    for column in OPTIONAL_COLUMNS:
        if column not in dishes:
            dishes[column] = ""

    input_rows = len(dishes)
    blank_names = dishes["dish_name"].astype(str).str.strip().eq("")
    dropped_blank_names = int(blank_names.sum())
    dishes = dishes.loc[~blank_names].copy()

    text_fallbacks = {
        "cuisine": UNAVAILABLE_CUISINE,
        "common_ingredients": UNAVAILABLE_INGREDIENTS,
        "allergens": UNAVAILABLE_ALLERGENS,
        "dietary_tags": "Dietary information unavailable",
        "description": "Description unavailable",
        "source": UNAVAILABLE_SOURCE,
    }
    dishes["dish_name"] = dishes["dish_name"].map(clean_text)
    for column, fallback in text_fallbacks.items():
        dishes[column] = dishes[column].map(lambda value, f=fallback: clean_text(value, f))
    for column in OPTIONAL_COLUMNS:
        dishes[column] = dishes[column].map(clean_text)

    dishes["normalized_dish_name"] = dishes["dish_name"].map(normalize_dish_name)
    dishes = dishes[dishes["normalized_dish_name"].ne("")].copy()
    dishes["source_restaurant_count"] = pd.to_numeric(
        dishes.get("source_restaurant_count", 0), errors="coerce"
    ).fillna(0).astype(int)

    scores = dishes.apply(quality_score, axis=1, result_type="expand")
    dishes[["_quality", "_optional_count", "_source_count"]] = scores
    dishes = dishes.sort_values(
        ["_quality", "_optional_count", "_source_count", "dish_id"],
        ascending=[False, False, False, True],
        kind="stable",
    )
    duplicate_count = int(dishes.duplicated("normalized_dish_name").sum())
    dishes = dishes.drop_duplicates("normalized_dish_name", keep="first")
    dishes = dishes.sort_values("dish_id", kind="stable").reset_index(drop=True)

    report = {
        "input_rows": input_rows,
        "canonical_rows": len(dishes),
        "dropped_blank_dish_names": dropped_blank_names,
        "removed_normalized_duplicates": duplicate_count,
        "missing_ingredients_after_fallback": int(
            dishes["common_ingredients"].eq(UNAVAILABLE_INGREDIENTS).sum()
        ),
        "missing_allergens_after_fallback": int(
            dishes["allergens"].eq(UNAVAILABLE_ALLERGENS).sum()
        ),
        "dataset_sha256": dataset_sha256(path),
    }
    return dishes, report


def display_list(value: str, fallback: str = "Not available") -> str:
    return value.replace(";", ", ") if value else fallback


def culinary_document(row: Mapping[str, Any]) -> str:
    """Convert one structured CSV row into readable text for embedding and storage."""

    allergens = str(row["allergens"])
    if allergens == "none_identified_from_name":
        allergens = "No allergen identified from the menu name; absence is not confirmed"

    return "\n".join([
        f"Dish: {row['dish_name']}",
        f"Cuisine: {row['cuisine']}",
        f"Description: {row['description']}",
        f"Common ingredients: {display_list(str(row['common_ingredients']), UNAVAILABLE_INGREDIENTS)}",
        f"Possible allergens: {display_list(allergens, UNAVAILABLE_ALLERGENS)}",
        f"Dietary tags: {display_list(str(row['dietary_tags']))}",
        f"Preparation method: {display_list(str(row.get('preparation_method', '')))}",
        f"Ingredient variations: {display_list(str(row.get('ingredient_variations', '')))}",
        f"Possible substitutions: {display_list(str(row.get('possible_substitutions', '')))}",
        f"Allergen confidence: {clean_text(row.get('allergen_confidence'), 'Unknown')}",
        f"Source: {row['source']}",
    ])


# -----------------------------------------------------------------------------
# 4. Persistent vector-index construction
# -----------------------------------------------------------------------------

def load_embedding_model():
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE_DIR))
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(MODEL_CACHE_DIR))
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL, cache_folder=str(MODEL_CACHE_DIR), device="cpu")


def chroma_client():
    import chromadb
    from chromadb.config import Settings

    VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(VECTOR_DB_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def load_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Build the RAG index first; missing {MANIFEST_PATH}")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def build_vector_index(
    *, rebuild: bool = False, distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD
) -> dict[str, Any]:
    """Build embeddings once, or reuse the index when its manifest still matches."""

    if not 0 < distance_threshold < 2:
        raise ValueError("distance_threshold must be between 0 and 2")

    dishes, validation = load_culinary_dataset()
    client = chroma_client()

    if MANIFEST_PATH.is_file() and not rebuild:
        manifest = load_manifest()
        unchanged = (
            manifest.get("dataset_sha256") == validation["dataset_sha256"]
            and manifest.get("embedding_model") == EMBEDDING_MODEL
            and manifest.get("canonical_rows") == len(dishes)
        )
        if unchanged and client.get_collection(COLLECTION_NAME).count() == len(dishes):
            return {**manifest, "build_status": "reused"}
        raise RuntimeError("Dataset or model changed; run an explicit index rebuild")

    existing_names = {collection.name for collection in client.list_collections()}
    if COLLECTION_NAME in existing_names:
        client.delete_collection(COLLECTION_NAME)
    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine", "embedding_model": EMBEDDING_MODEL},
    )

    records = dishes.to_dict(orient="records")
    documents = [culinary_document(row) for row in records]
    embeddings = load_embedding_model().encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    for start in range(0, len(records), 200):
        stop = min(start + 200, len(records))
        batch = records[start:stop]
        collection.add(
            ids=[str(row["dish_id"]) for row in batch],
            documents=documents[start:stop],
            embeddings=embeddings[start:stop].tolist(),
            metadatas=[{
                "dish_id": str(row["dish_id"]),
                "dish_name": str(row["dish_name"]),
                "normalized_dish_name": str(row["normalized_dish_name"]),
                "cuisine": str(row["cuisine"]),
                "source": str(row["source"]),
            } for row in batch],
        )

    manifest = {
        "collection_name": COLLECTION_NAME,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dimensions": int(embeddings.shape[1]),
        "distance_metric": "cosine",
        "distance_threshold": float(distance_threshold),
        "dataset_sha256": validation["dataset_sha256"],
        "canonical_rows": len(dishes),
        "normalizer_version": 1,
        "document_schema_version": 1,
        "query_schema_version": QUERY_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation": validation,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {**manifest, "build_status": "rebuilt"}


# -----------------------------------------------------------------------------
# 5. Runtime exact-first retrieval
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RetrievalResult:
    requested_dish: str
    normalized_query: str
    matched_dish: str | None
    matched_dish_id: str | None
    nearest_dish: str | None
    match_type: str
    retrieval_status: str
    distance: float | None
    confidence: float | None
    cuisine: str
    common_ingredients: str
    allergens: str
    dietary_tags: str
    description: str
    possible_substitutions: str
    preparation_method: str
    ingredient_variations: str
    allergen_confidence: str
    source: str
    requires_manual_confirmation: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CulinaryRAG:
    """Retrieve exact matches first and use vector search only as a fallback."""

    def __init__(self, *, distance_threshold: float | None = None) -> None:
        self.frame, validation = load_culinary_dataset()
        records = self.frame.to_dict(orient="records")
        self.records_by_normalized = {row["normalized_dish_name"]: row for row in records}
        self.records_by_id = {str(row["dish_id"]): row for row in records}

        self.manifest = load_manifest()
        if self.manifest.get("dataset_sha256") != validation["dataset_sha256"]:
            raise RuntimeError("Dataset changed after indexing; rebuild the RAG index")
        if self.manifest.get("embedding_model") != EMBEDDING_MODEL:
            raise RuntimeError("Embedding model changed; rebuild the RAG index")
        if self.manifest.get("query_schema_version") != QUERY_SCHEMA_VERSION:
            raise RuntimeError("Query configuration does not match the index manifest")

        self.distance_threshold = float(
            distance_threshold
            if distance_threshold is not None
            else self.manifest["distance_threshold"]
        )
        self.client = chroma_client()
        self.collection = self.client.get_collection(COLLECTION_NAME)
        if self.collection.count() != len(self.frame):
            raise RuntimeError("Chroma row count does not match the culinary dataset")
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = load_embedding_model()
        return self._model

    def semantic_candidates(
        self, query: str, *, cuisine: str | None = None, top_k: int = 3
    ) -> list[dict[str, Any]]:
        """Return the nearest semantic candidates without accepting any of them."""

        if not 1 <= top_k <= 20:
            raise ValueError("top_k must be between 1 and 20")
        if not normalize_dish_name(query):
            return []

        vector = self.model.encode(
            [semantic_query_text(query)], normalize_embeddings=True, convert_to_numpy=True
        )[0].tolist()
        response = self.collection.query(
            query_embeddings=[vector],
            n_results=min(top_k, self.collection.count()),
            where=cuisine_filter(cuisine),
            include=["metadatas", "distances"],
        )
        return [{
            "dish_id": dish_id,
            "dish_name": metadata["dish_name"],
            "cuisine": metadata["cuisine"],
            "distance": round(float(distance), 6),
            "similarity": round(max(0.0, 1.0 - float(distance)), 6),
        } for dish_id, metadata, distance in zip(
            response.get("ids", [[]])[0],
            response.get("metadatas", [[]])[0],
            response.get("distances", [[]])[0],
        )]

    def retrieve(
        self, requested_dish: str, *, cuisine: str | None = None, top_k: int = 3
    ) -> RetrievalResult:
        """Return a safe structured exact, vector, or unavailable result."""

        normalized = normalize_dish_name(requested_dish)
        if not normalized:
            return self._unavailable(requested_dish, normalized, "no_reliable_match")

        exact = self.records_by_normalized.get(normalized)
        if exact:
            return self._matched(requested_dish, normalized, exact, "exact", 0.0)

        candidates = self.semantic_candidates(requested_dish, cuisine=cuisine, top_k=top_k)
        if not candidates:
            return self._unavailable(requested_dish, normalized, "no_reliable_match")

        nearest = candidates[0]
        distance = float(nearest["distance"])
        if distance <= self.distance_threshold:
            record = self.records_by_id[str(nearest["dish_id"])]
            return self._matched(requested_dish, normalized, record, "vector", distance)

        status = (
            "low_confidence"
            if distance <= self.distance_threshold + LOW_CONFIDENCE_MARGIN
            else "no_reliable_match"
        )
        return self._unavailable(
            requested_dish,
            normalized,
            status,
            nearest_dish=str(nearest["dish_name"]),
            distance=distance,
        )

    def retrieve_many(
        self, candidates: Sequence[str | Mapping[str, Any]], *, top_k: int = 3
    ) -> list[RetrievalResult]:
        """Retrieve knowledge for the 5-10 rows produced by restaurant filtering."""

        results = []
        for candidate in candidates:
            if isinstance(candidate, Mapping):
                name = str(candidate.get("dish_name", ""))
                cuisine = clean_text(candidate.get("cuisine")) or None
            else:
                name, cuisine = str(candidate), None
            results.append(self.retrieve(name, cuisine=cuisine, top_k=top_k))
        return results

    @staticmethod
    def _matched(
        requested: str,
        normalized: str,
        row: Mapping[str, Any],
        match_type: str,
        distance: float,
    ) -> RetrievalResult:
        return RetrievalResult(
            requested_dish=requested,
            normalized_query=normalized,
            matched_dish=str(row["dish_name"]),
            matched_dish_id=str(row["dish_id"]),
            nearest_dish=str(row["dish_name"]),
            match_type=match_type,
            retrieval_status="matched",
            distance=round(distance, 6),
            confidence=round(max(0.0, 1.0 - distance), 6),
            cuisine=str(row["cuisine"]),
            common_ingredients=str(row["common_ingredients"]),
            allergens=str(row["allergens"]),
            dietary_tags=str(row["dietary_tags"]),
            description=str(row["description"]),
            possible_substitutions=str(row.get("possible_substitutions", "")),
            preparation_method=str(row.get("preparation_method", "")),
            ingredient_variations=str(row.get("ingredient_variations", "")),
            allergen_confidence=str(row.get("allergen_confidence", "")),
            source=str(row["source"]),
        )

    @staticmethod
    def _unavailable(
        requested: str,
        normalized: str,
        status: str,
        *,
        nearest_dish: str | None = None,
        distance: float | None = None,
    ) -> RetrievalResult:
        confidence = max(0.0, 1.0 - distance) if distance is not None else None
        return RetrievalResult(
            requested_dish=requested,
            normalized_query=normalized,
            matched_dish=None,
            matched_dish_id=None,
            nearest_dish=nearest_dish,
            match_type="none",
            retrieval_status=status,
            distance=round(distance, 6) if distance is not None else None,
            confidence=round(confidence, 6) if confidence is not None else None,
            cuisine=UNAVAILABLE_CUISINE,
            common_ingredients=UNAVAILABLE_INGREDIENTS,
            allergens=UNAVAILABLE_ALLERGENS,
            dietary_tags="Dietary information unavailable",
            description="Culinary information unavailable",
            possible_substitutions="",
            preparation_method="",
            ingredient_variations="",
            allergen_confidence="unknown",
            source=UNAVAILABLE_SOURCE,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Build or reuse the index")
    parser.add_argument("--rebuild", action="store_true", help="Explicitly rebuild embeddings")
    parser.add_argument("--query", help="Retrieve knowledge for one dish")
    parser.add_argument("--cuisine", help="Optional cuisine metadata filter")
    args = parser.parse_args()

    if args.build or args.rebuild:
        print(json.dumps(build_vector_index(rebuild=args.rebuild), indent=2))
    if args.query:
        result = CulinaryRAG().retrieve(args.query, cuisine=args.cuisine)
        print(json.dumps(result.to_dict(), indent=2))
    if not (args.build or args.rebuild or args.query):
        parser.print_help()


if __name__ == "__main__":
    main()
