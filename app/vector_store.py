import json
from pathlib import Path

import faiss
import numpy as np


MODELS_DIR = Path("models")
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.npy"
FAISS_INDEX_PATH = MODELS_DIR / "faiss_index.bin"
METADATA_PATH = MODELS_DIR / "embedding_metadata.jsonl"
INDEX_INFO_PATH = MODELS_DIR / "faiss_index_metadata.json"


def build_faiss_index() -> None:
    """Build and persist a FAISS index from saved embeddings."""
    if not EMBEDDINGS_PATH.exists():
        print("Embeddings file not found. Run embedding_generation.py first.")
        return
    if not METADATA_PATH.exists():
        print("Embedding metadata file not found. Run embedding_generation.py first.")
        return

    embeddings = np.load(EMBEDDINGS_PATH).astype("float32")
    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    metadata_lines = METADATA_PATH.read_text(encoding="utf-8").splitlines()
    INDEX_INFO_PATH.write_text(
        json.dumps(
            {
                "index_type": "IndexFlatL2",
                "vector_count": int(embeddings.shape[0]),
                "dimension": int(dimension),
                "metadata_records": len(metadata_lines),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved FAISS index to {FAISS_INDEX_PATH}")


if __name__ == "__main__":
    build_faiss_index()
