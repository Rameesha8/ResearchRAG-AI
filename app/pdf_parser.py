import json
from pathlib import Path

import fitz


RAW_PDF_DIR = Path("data/raw_pdfs")
EXTRACTED_TEXT_DIR = Path("data/extracted_text")
PROCESSED_TEXT_DIR = Path("data/processed_text")


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from a PDF file using PyMuPDF."""
    with fitz.open(pdf_path) as document:
        pages = [page.get_text("text") for page in document]
    return "\n".join(pages).strip()


def parse_all_pdfs() -> None:
    """Parse all PDFs in the raw data folder and save text plus metadata."""
    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_paths = sorted(RAW_PDF_DIR.rglob("*.pdf"))

    if not pdf_paths:
        print("No PDF files found in data/raw_pdfs.")
        return

    for pdf_path in pdf_paths:
        extracted_text = extract_text_from_pdf(pdf_path)
        text_output_path = EXTRACTED_TEXT_DIR / f"{pdf_path.stem}.txt"
        processed_output_path = PROCESSED_TEXT_DIR / f"{pdf_path.stem}.txt"
        metadata_output_path = EXTRACTED_TEXT_DIR / f"{pdf_path.stem}.json"

        text_output_path.write_text(extracted_text, encoding="utf-8")
        processed_output_path.write_text(extracted_text, encoding="utf-8")
        metadata = {
            "source_file": pdf_path.name,
            "source_path": str(pdf_path),
            "text_file": str(text_output_path),
        }
        metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved extracted text to {text_output_path}")


if __name__ == "__main__":
    parse_all_pdfs()
