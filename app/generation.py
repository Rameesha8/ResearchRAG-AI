import os
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv

try:
    from .fallback_models import predict_with_local_models
except ImportError:
    from fallback_models import predict_with_local_models

try:
    from google import genai
except ImportError:
    genai = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def format_citation(record: dict) -> str:
    """Build a compact citation string for a retrieved chunk."""
    return (
        f"[arXiv:{record.get('document_id', 'unknown')} | "
        f"{record.get('title', 'Untitled')} | "
        f"{record.get('authors', 'Unknown authors')}]"
    )


def build_context(records: Iterable[dict]) -> str:
    """Format retrieval results into a model-friendly context block."""
    context_parts = []
    for index, record in enumerate(records, start=1):
        context_parts.append(
            "\n".join(
                [
                    f"Source {index}",
                    f"Citation: {format_citation(record)}",
                    f"Categories: {record.get('categories', 'Unknown categories')}",
                    f"Updated: {record.get('update_date', 'Unknown date')}",
                    f"Content: {record.get('text', '')}",
                ]
            )
        )
    return "\n\n".join(context_parts)


def build_fallback_answer(question: str, records: list[dict]) -> str:
    """Create a retrieval-grounded answer enhanced by trained local models."""
    if not records:
        return "I could not find relevant context for that question in the indexed arXiv subset."

    predictions = predict_with_local_models(question)
    opening = [
        f"Based on the retrieved arXiv context, the strongest evidence for '{question}' comes from {len(records)} relevant chunk(s)."
    ]

    if predictions:
        opening.append("Local model predictions for the query topic:")
        for model_name, details in predictions.items():
            opening.append(
                f"- {model_name}: {details['label']} (confidence {details['confidence']:.2%})"
            )

    evidence_lines = ["Supporting evidence:"]
    for record in records[:3]:
        snippet = record.get("text", "").strip().replace("\n", " ")
        snippet = snippet[:320].rstrip()
        evidence_lines.append(f"- {snippet} {format_citation(record)}")

    closing = "This answer was produced without Gemini and uses trained local classifiers plus retrieval evidence."
    return "\n".join([*opening, *evidence_lines, closing])


def build_system_prompt() -> str:
    """Create the instruction prompt for grounded Gemini answers."""
    return (
        "You are a research assistant answering questions about academic papers. "
        "Use only the provided context. If the context is insufficient, say that clearly. "
        "Write a concise answer in plain academic English with inline citations using the exact citation strings provided. "
        "Do not invent claims, methods, or results."
    )


def generate_answer(question: str, records: list[dict]) -> tuple[str, str]:
    """Generate a grounded answer using Gemini if configured, else use trained fallback synthesis."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or genai is None:
        return build_fallback_answer(question, records), "fallback"

    try:
        client = genai.Client(api_key=api_key)
        context = build_context(records)
        prompt = (
            f"{build_system_prompt()}\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}"
        )
        response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        answer_text = getattr(response, "text", None) or build_fallback_answer(question, records)
        return answer_text, "gemini"
    except Exception as exc:
        fallback = build_fallback_answer(question, records)
        return f"{fallback}\n\nNote: Gemini request failed: {exc}", "fallback-error"
