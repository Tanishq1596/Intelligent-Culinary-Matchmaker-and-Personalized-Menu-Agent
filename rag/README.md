# Culinary RAG

This folder contains the exact-first culinary retrieval subsystem. Restaurant
filtering should supply only its final 5-10 dish candidates.

## Interview walkthrough

Explain `rag_pipeline.py` in the same five sections used by the source code:

1. **Configuration:** define the CSV, model, Chroma directory, and threshold.
2. **Normalization:** remove serving text and expand common Indian food terms.
3. **Data preparation:** validate fields, add explicit fallbacks, and deduplicate
   normalized dish names by record quality.
4. **Index construction:** convert rows to readable documents, embed them once,
   persist them in Chroma, and record the dataset hash in a manifest.
5. **Runtime retrieval:** try exact matching first, use semantic search only as a
   fallback, accept distances up to `0.35`, and withhold culinary fields for weak
   matches.

The central interview point is that RAG retrieves knowledge; it does not determine
medical safety. Every result still requires deterministic safety rules and
restaurant confirmation.

## Retrieval order

1. Normalize the candidate dish name.
2. Return a normalized exact match when available.
3. Otherwise query the persistent Chroma collection with
   `sentence-transformers/all-MiniLM-L6-v2`.
4. Apply the calibrated cosine-distance threshold.
5. Return structured unavailable fields when confidence is insufficient.

The vector database is stored in `../vector_db/`. The manifest records the
culinary CSV hash, embedding model, vector dimensions, metric, threshold, and
canonical row count. Runtime initialization validates this manifest and never
rebuilds embeddings automatically.

## Commands

Build or reuse an unchanged index:

```powershell
.\.tools\python312\python.exe .\culinary_matchmaker\rag\rag_pipeline.py --build
```

Explicitly rebuild after changing the dataset or embedding model:

```powershell
.\.tools\python312\python.exe .\culinary_matchmaker\rag\rag_pipeline.py --rebuild
```

Query one dish:

```powershell
.\.tools\python312\python.exe .\culinary_matchmaker\rag\rag_pipeline.py --query "Special Dal Tadka Full"
```

Run the smoke tests:

```powershell
.\.tools\python312\python.exe .\culinary_matchmaker\rag\test_rag_pipeline.py
```

Run the labeled retrieval evaluation:

```powershell
.\.tools\python312\python.exe .\culinary_matchmaker\rag\evaluate_rag.py
```

This writes `retrieval_evaluation.json` beside the Python code. No notebook is
required for building, querying, testing, or evaluating the RAG system.

All matched records still require restaurant confirmation. A low-confidence or
unmatched result deliberately returns unavailable ingredient and allergen fields;
the closest vector result is never treated as established culinary knowledge.
