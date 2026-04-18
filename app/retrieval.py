import json
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.npy"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
METADATA_PATH = MODELS_DIR / "embedding_metadata.jsonl"


@lru_cache(maxsize=1)
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


@lru_cache(maxsize=1)
def load_embedding_model() -> SentenceTransformer:
    """Load the sentence-transformer once per process."""
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def load_faiss_index() -> faiss.Index:
    """Load the persisted FAISS index once per process."""
    return faiss.read_index(str(FAISS_INDEX_PATH))


def retrieve_similar_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Return the most similar chunks for a user query."""
    if not query.strip():
        return []
    if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
        return []

    metadata = load_metadata()
    if not metadata:
        return []

    model = load_embedding_model()
    index = load_faiss_index()
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
