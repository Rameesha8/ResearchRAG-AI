import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


CHUNKS_DIR = Path("data/chunks")
MODELS_DIR = Path("models")
EMBEDDINGS_PATH = MODELS_DIR / "embeddings.npy"
METADATA_PATH = MODELS_DIR / "embedding_metadata.jsonl"


def load_chunk_records() -> tuple[list[str], list[dict]]:
    """Load chunked documents and their metadata records."""
    documents: list[str] = []
    records: list[dict] = []

    for chunk_path in sorted(CHUNKS_DIR.glob("*.jsonl")):
        for line in chunk_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            documents.append(record["text"])
            records.append(record)

    return documents, records


def generate_embeddings(model_name: str = "all-MiniLM-L6-v2") -> None:
    """Generate sentence-transformer embeddings for chunked documents."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    documents, records = load_chunk_records()

    if not documents:
        print("No chunked text files found.")
        return

    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, convert_to_numpy=True, show_progress_bar=True)

    np.save(EMBEDDINGS_PATH, embeddings)
    with METADATA_PATH.open("w", encoding="utf-8") as output_file:
        for record in records:
            output_file.write(json.dumps(record) + "\n")
    print(f"Saved {len(records)} embeddings to {EMBEDDINGS_PATH}")


if __name__ == "__main__":
    generate_embeddings()
