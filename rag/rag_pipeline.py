"""Build and query the culinary knowledge base with embeddings and ChromaDB."""

from pathlib import Path
import re

import chromadb
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "culinary_knowledge.csv"
VECTOR_DB_PATH = PROJECT_ROOT / "vector_db"
MODEL_CACHE_PATH = PROJECT_ROOT / ".model_cache"

COLLECTION_NAME = "culinary_dishes_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DISTANCE_THRESHOLD = 0.35


CULINARY_SYNONYMS = [
    (("grilled cottage cheese",), "paneer tikka"),
    (("potato cauliflower", "potato and cauliflower"), "aloo gobi masala"),
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
    (("yogurt cucumber", "cucumber yogurt"), "cucumber raita"),
    (("flatbread",), "roti chapati"),
    (("rice lentil pancake", "rice and lentil pancake"), "dosa"),
    (("steamed rice lentil cake", "steamed rice and lentil cake"), "idli"),
    (("flattened rice",), "poha"),
    (("semolina breakfast",), "upma"),
]


def normalize_dish_name(dish_name):
    text = str(dish_name).casefold()
    bracketed_parts = re.findall(r"\([^)]*\)|\[[^]]*\]|\{[^}]*\}", text)
    serving_words = ("full", "half", "serve", "pc", "piece", "ml", "gm", "kg", "pack")
    for part in bracketed_parts:
        if any(character.isdigit() for character in part) or any(
            word in part for word in serving_words
        ):
            text = text.replace(part, " ")
    text = re.sub(
        r"\b\d+(?:\.\d+)?\s*(?:ml|l|ltr|litre|g|gm|kg|pc|pcs|piece|pieces|serve|serves)\b",
        " ",
        text,
    )
    text = re.sub(r"\b(?:special|full|half|combo)\b", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def load_knowledge_base():
    dishes = pd.read_csv(DATA_PATH, keep_default_na=False)
    dishes["normalized_dish_name"] = dishes["dish_name"].map(normalize_dish_name)

    # When menu variants normalize to the same name, keep the row with the
    # strongest allergen information and then the most restaurant support.
    confidence_order = {"high": 3, "medium": 2, "low": 1}
    dishes["_confidence"] = dishes["allergen_confidence"].map(confidence_order)
    dishes = dishes.sort_values(
        ["_confidence", "source_restaurant_count", "dish_id"],
        ascending=[False, False, True],
    )
    dishes = dishes.drop_duplicates("normalized_dish_name")
    return dishes.drop(columns="_confidence")


def create_document(row):
    return "\n".join([
        f"Dish: {row['dish_name']}",
        f"Cuisine: {row['cuisine']}",
        f"Description: {row['description']}",
        f"Common ingredients: {row['common_ingredients']}",
        f"Possible allergens: {row['allergens']}",
        f"Dietary tags: {row['dietary_tags']}",
        f"Preparation method: {row['preparation_method']}",
        f"Ingredient variations: {row['ingredient_variations']}",
        f"Possible substitutions: {row['possible_substitutions']}",
        f"Allergen confidence: {row['allergen_confidence'] or 'Unknown'}",
        f"Source: {row['source']}",
    ])


def load_embedding_model():
    from sentence_transformers import SentenceTransformer

    MODEL_CACHE_PATH.mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(
        EMBEDDING_MODEL,
        cache_folder=str(MODEL_CACHE_PATH),
        device="cpu",
    )


def get_chroma_client():
    VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(VECTOR_DB_PATH))


def build_vector_index(rebuild=False):
    dishes = load_knowledge_base()
    records = dishes.to_dict(orient="records")

    documents = [create_document(row) for row in records]
    document_ids = [str(row["dish_id"]) for row in records]
    metadata = [
        {
            "dish_id": str(row["dish_id"]),
            "dish_name": row["dish_name"],
            "normalized_dish_name": row["normalized_dish_name"],
            "cuisine": row["cuisine"],
            "source": row["source"],
        }
        for row in records
    ]

    client = get_chroma_client()
    collection_names = [item.name for item in client.list_collections()]

    if COLLECTION_NAME in collection_names and not rebuild:
        collection = client.get_collection(COLLECTION_NAME)
        if collection.count() == len(documents):
            return {"status": "reused", "documents": len(documents)}

    if COLLECTION_NAME in collection_names:
        client.delete_collection(COLLECTION_NAME)

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    embeddings = load_embedding_model().encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    collection.add(
        ids=document_ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadata,
    )

    return {
        "status": "rebuilt",
        "documents": len(documents),
        "embedding_dimensions": len(embeddings[0]),
    }


def cuisine_filter(cuisine):
    cuisines = [item.strip() for item in re.split(r"[,/]", cuisine) if item.strip()]
    if len(cuisines) == 1:
        return {"cuisine": cuisines[0]}
    return {"cuisine": {"$in": cuisines}}


def prepare_query(dish_name):
    normalized = normalize_dish_name(dish_name)
    synonyms = [
        replacement
        for phrases, replacement in CULINARY_SYNONYMS
        if any(phrase in normalized for phrase in phrases)
    ]
    query = f"Dish: {dish_name}"
    if synonyms:
        query += f"\nAlternative Indian dish names: {', '.join(synonyms)}"
    return query


class CulinaryRAG:
    def __init__(self, distance_threshold=DISTANCE_THRESHOLD):
        self.frame = load_knowledge_base()
        records = self.frame.to_dict(orient="records")
        self.records_by_name = {
            row["normalized_dish_name"]: row for row in records
        }
        self.records_by_id = {str(row["dish_id"]): row for row in records}
        self.distance_threshold = distance_threshold
        self.collection = get_chroma_client().get_collection(COLLECTION_NAME)
        self._embed_model = None

    # The agent passes all dishes shortlisted by Pandas filtering here.
    def retrieve_many(self, candidates, top_k=3):
        return [
            self.retrieve(candidate["dish_name"], candidate["cuisine"], top_k)
            for candidate in candidates
        ]

    def retrieve(self, requested_dish, cuisine, top_k=3):
        normalized = normalize_dish_name(requested_dish)
        exact_match = self.records_by_name.get(normalized)

        if exact_match:
            return self.matched_result(requested_dish, exact_match)

        candidates = self.semantic_candidates(requested_dish, cuisine, top_k)
        if candidates and candidates[0]["distance"] <= self.distance_threshold:
            row = self.records_by_id[candidates[0]["dish_id"]]
            return self.matched_result(requested_dish, row)

        return {
            "requested_dish": requested_dish,
            "matched_dish": None,
            "matched_dish_id": None,
        }

    def semantic_candidates(self, dish_name, cuisine, top_k=3):
        dish_embedding = self.embed_model.encode(
            [prepare_query(dish_name)],
            normalize_embeddings=True,
        )[0]

        search_results = self.collection.query(
            query_embeddings=[dish_embedding.tolist()],
            n_results=top_k,
            where=cuisine_filter(cuisine),
            include=["metadatas", "distances"],
        )

        candidates = []
        for dish_id, metadata, distance in zip(
            search_results["ids"][0],
            search_results["metadatas"][0],
            search_results["distances"][0],
        ):
            candidates.append({
                "dish_id": dish_id,
                "dish_name": metadata["dish_name"],
                "distance": round(float(distance), 6),
            })
        return candidates

    def matched_result(self, requested_dish, row):
        return {
            "requested_dish": requested_dish,
            "matched_dish": row["dish_name"],
            "matched_dish_id": str(row["dish_id"]),
            "cuisine": row["cuisine"],
            "common_ingredients": row["common_ingredients"],
            "allergens": row["allergens"],
            "dietary_tags": row["dietary_tags"],
            "description": row["description"],
            "possible_substitutions": row["possible_substitutions"],
            "preparation_method": row["preparation_method"],
            "ingredient_variations": row["ingredient_variations"],
            "allergen_confidence": row["allergen_confidence"],
            "source": row["source"],
            "requires_manual_confirmation": True,
        }

    @property
    def embed_model(self):
        if self._embed_model is None:
            self._embed_model = load_embedding_model()
        return self._embed_model
