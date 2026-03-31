import json
import re
from pathlib import Path


PROCESSED_TEXT_DIR = Path("data/processed_text")
ARXIV_DOCUMENTS_PATH = PROCESSED_TEXT_DIR / "arxiv_documents.jsonl"
CHUNKS_DIR = Path("data/chunks")
CHUNKS_OUTPUT_PATH = CHUNKS_DIR / "arxiv_chunks.jsonl"


def clean_text(text: str) -> str:
    """Perform basic whitespace cleanup while preserving readable content."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into overlapping character chunks for retrieval."""
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, 0)

    return [chunk for chunk in chunks if chunk]


def build_document_text(record: dict) -> str:
    """Build the retrieval text from normalized arXiv metadata."""
    title = clean_text(record.get("title", ""))
    abstract = clean_text(record.get("abstract", ""))
    categories = clean_text(record.get("categories", ""))
    authors = clean_text(record.get("authors", ""))

    parts = [
        f"Title: {title}",
        f"Authors: {authors}",
        f"Categories: {categories}",
        f"Abstract: {abstract}",
    ]
    return "\n".join(part for part in parts if part.strip())


def preprocess_arxiv_documents() -> None:
    """Clean normalized arXiv documents and export chunk records as JSONL."""
    if not ARXIV_DOCUMENTS_PATH.exists():
        print(f"Normalized arXiv file not found: {ARXIV_DOCUMENTS_PATH}")
        return

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    chunk_count = 0
    document_count = 0

    with ARXIV_DOCUMENTS_PATH.open("r", encoding="utf-8") as input_file, CHUNKS_OUTPUT_PATH.open(
        "w", encoding="utf-8"
    ) as output_file:
        for line in input_file:
            if not line.strip():
                continue

            record = json.loads(line)
            document_count += 1
            document_text = build_document_text(record)
            chunks = chunk_text(document_text)

            for index, chunk in enumerate(chunks):
                chunk_record = {
                    "chunk_id": f"{record['document_id']}-{index}",
                    "document_id": record["document_id"],
                    "source": record.get("source", "arxiv"),
                    "title": clean_text(record.get("title", "")),
                    "authors": clean_text(record.get("authors", "")),
                    "categories": clean_text(record.get("categories", "")),
                    "update_date": record.get("update_date"),
                    "chunk_index": index,
                    "text": chunk,
                }
                output_file.write(json.dumps(chunk_record) + "\n")
                chunk_count += 1

    print(f"Processed {document_count} arXiv records into {chunk_count} chunks")


if __name__ == "__main__":
    preprocess_arxiv_documents()
