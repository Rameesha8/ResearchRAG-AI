import json
import os
from pathlib import Path


ARXIV_DATASET_PATH = Path("data/datasets/arxiv/arxiv-metadata-oai-snapshot.json")
PROCESSED_TEXT_DIR = Path("data/processed_text")
NORMALIZED_OUTPUT_PATH = PROCESSED_TEXT_DIR / "arxiv_documents.jsonl"
DEFAULT_MAX_RECORDS = int(os.getenv("ARXIV_MAX_RECORDS", "5000"))


def normalize_arxiv_record(record: dict) -> dict:
    """Convert a raw arXiv metadata record into a normalized project schema."""
    return {
        "document_id": record.get("id", ""),
        "title": (record.get("title") or "").strip(),
        "abstract": (record.get("abstract") or "").strip(),
        "authors": (record.get("authors") or "").strip(),
        "categories": (record.get("categories") or "").strip(),
        "update_date": record.get("update_date"),
        "source": "arxiv",
    }


def export_arxiv_documents(max_records: int = DEFAULT_MAX_RECORDS) -> None:
    """Stream the arXiv snapshot file and export normalized JSONL documents."""
    if not ARXIV_DATASET_PATH.exists():
        print(f"Dataset file not found: {ARXIV_DATASET_PATH}")
        return

    PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    exported = 0

    with ARXIV_DATASET_PATH.open("r", encoding="utf-8") as source_file, NORMALIZED_OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line in source_file:
            if not line.strip():
                continue

            raw_record = json.loads(line)
            normalized_record = normalize_arxiv_record(raw_record)

            if not normalized_record["title"] or not normalized_record["abstract"]:
                continue

            output_file.write(json.dumps(normalized_record) + "\n")
            exported += 1

            if exported >= max_records:
                break

    print(f"Exported {exported} arXiv records to {NORMALIZED_OUTPUT_PATH}")


if __name__ == "__main__":
    export_arxiv_documents()
