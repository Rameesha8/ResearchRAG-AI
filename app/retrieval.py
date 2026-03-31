import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
MODELS_DIR = Path("models")
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.npy"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
METADATA_PATH = MODELS_DIR / "embedding_metadata.jsonl"


def load_metadata() -> list[dict]:
    """Load saved chunk metadata records."""
    if not METADATA_PATH.exists():
        return []

    records = []
    with METADATA_PATH.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if line.strip():
                records.append(json.loads(line))
    return records


def retrieve_similar_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Return the most similar chunks for a user query."""
    if not query.strip():
        return []
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        return []

    metadata = load_metadata()
    if not metadata:
        return []

    model = SentenceTransformer(MODEL_NAME)
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    query_vector = model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(query_vector, top_k)

    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        record = metadata[idx].copy()
        record["score"] = float(score)
        results.append(record)

    return results
