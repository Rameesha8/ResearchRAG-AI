import json
import os
import subprocess
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

try:
    from .fallback_models import predict_with_local_models
    from .generation import generate_answer
    from .retrieval import retrieve_similar_chunks
except ImportError:
    from fallback_models import predict_with_local_models
    from generation import generate_answer
    from retrieval import retrieve_similar_chunks


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)

ARXIV_DATASET_PATH = PROJECT_ROOT / "data/datasets/arxiv/arxiv-metadata-oai-snapshot.json"
NORMALIZED_DOCS_PATH = PROJECT_ROOT / "data/processed_text/arxiv_documents.jsonl"
FAISS_INDEX_PATH = PROJECT_ROOT / "models/faiss_index.bin"
METADATA_PATH = PROJECT_ROOT / "models/embedding_metadata.jsonl"
LOGREG_MODEL_PATH = PROJECT_ROOT / "models/logistic_regression_pipeline.pkl"
MLP_MODEL_PATH = PROJECT_ROOT / "models/mlp_classifier_pipeline.pkl"
METRICS_PATH = PROJECT_ROOT / "reports/model_metrics.json"
EDA_DIR = PROJECT_ROOT / "reports/eda_exports"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def run_step(script_name: str) -> tuple[bool, str]:
    """Run one pipeline step with the current Python interpreter."""
    script_path = APP_DIR / script_name
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def load_metrics() -> dict | None:
    """Load saved training metrics if present."""
    if not METRICS_PATH.exists():
        return None
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def render_context_results(results: list[dict]) -> None:
    """Render retrieved context cleanly for presentation."""
    if not results:
        st.info("No retrieval results were found.")
        return

    for index, result in enumerate(results, start=1):
        with st.expander(f"Source {index}: {result.get('title', 'Untitled')}", expanded=index == 1):
            st.caption(
                f"Citation: arXiv:{result.get('document_id', 'Unknown')} | "
                f"Authors: {result.get('authors', 'Unknown authors')} | "
                f"Categories: {result.get('categories', 'Unknown categories')} | "
                f"Updated: {result.get('update_date', 'Unknown date')}"
            )
            st.write(f"Similarity score: {result.get('score', 0.0):.4f}")
            st.write(result.get("text", ""))


st.set_page_config(page_title="ResearchRAG AI", layout="wide")

st.title("ResearchRAG AI")
st.write(
    "An arXiv-based academic assistant for category prediction and citation-grounded question answering."
)

hero_col1, hero_col2, hero_col3 = st.columns(3)
hero_col1.metric("Retrieval Index", "Ready" if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists() else "Pending")
hero_col2.metric("ML Model", "Ready" if LOGREG_MODEL_PATH.exists() else "Pending")
hero_col3.metric("DL Model", "Ready" if MLP_MODEL_PATH.exists() else "Pending")

if os.getenv("GEMINI_API_KEY"):
    st.success(f"Gemini mode enabled with model `{GEMINI_MODEL}`")
else:
    st.info("Gemini key not found. The app will use the trained local fallback mode.")

main_tab1, main_tab2, main_tab3 = st.tabs(["Research Q&A", "Category Prediction", "Model Summary"])

with main_tab1:
    st.subheader("Ask About the Corpus")
    question = st.text_input("Enter a research question")
    result_count = st.slider("Retrieved sources", min_value=3, max_value=10, value=5, key="qa_slider")

    if question:
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            st.warning("The vector store is not ready yet. Run the indexing pipeline first.")
        else:
            with st.spinner("Retrieving relevant chunks and generating an answer..."):
                results = retrieve_similar_chunks(question, top_k=result_count)
                answer, mode = generate_answer(question, results)

            st.subheader("Answer")
            st.write(answer)
            st.caption(f"Answer mode: {mode}")
            st.subheader("Supporting Sources")
            render_context_results(results)

with main_tab2:
    st.subheader("Predict Paper Category")
    paper_text = st.text_area(
        "Paste a paper title and/or abstract",
        height=220,
    )
    if st.button("Predict Category") and paper_text.strip():
        local_predictions = predict_with_local_models(paper_text)
        if local_predictions:
            pred_col1, pred_col2 = st.columns(2)
            if "logistic_regression" in local_predictions:
                details = local_predictions["logistic_regression"]
                pred_col1.metric("Logistic Regression", details["label"])
                pred_col1.caption(f"Confidence: {details['confidence']:.2%}")
            if "mlp_classifier" in local_predictions:
                details = local_predictions["mlp_classifier"]
                pred_col2.metric("MLP Classifier", details["label"])
                pred_col2.caption(f"Confidence: {details['confidence']:.2%}")
        else:
            st.warning("Trained local models are not ready yet. Run model training first.")

with main_tab3:
    st.subheader("Current Model Performance")
    metrics = load_metrics()
    if metrics:
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.metric("LogReg Macro F1", f"{metrics['logistic_regression']['f1_macro']:.3f}")
        metric_col2.metric("MLP Macro F1", f"{metrics['mlp_classifier']['f1_macro']:.3f}")
        st.write("The MLP classifier is currently the stronger local category model on the top-10-category arXiv subset.")
    else:
        st.info("Training metrics are not available yet.")

with st.sidebar:
    st.header("Project Controls")
    st.caption("Use these controls during development or before a demo.")

    status_items = {
        "Dataset": ARXIV_DATASET_PATH.exists(),
        "Normalized docs": NORMALIZED_DOCS_PATH.exists(),
        "FAISS index": FAISS_INDEX_PATH.exists() and METADATA_PATH.exists(),
        "ML model": LOGREG_MODEL_PATH.exists(),
        "DL model": MLP_MODEL_PATH.exists(),
        "EDA exports": EDA_DIR.exists(),
    }
    for label, ok in status_items.items():
        st.write(f"{'OK' if ok else '...'} {label}")

    st.divider()
    st.subheader("Run Pipeline")
    sidebar_actions = [
        ("Run Loader", "arxiv_loader.py"),
        ("Run Preprocessing", "preprocessing.py"),
        ("Run Embeddings", "embedding_generation.py"),
        ("Run Vector Store", "vector_store.py"),
        ("Train Models", "model_training.py"),
        ("Export EDA", "export_eda.py"),
    ]

    for button_label, script_name in sidebar_actions:
        if st.button(button_label, use_container_width=True):
            with st.spinner(f"Running {script_name}..."):
                success, output = run_step(script_name)
            if success:
                st.success(f"{script_name} completed successfully.")
            else:
                st.error(f"{script_name} failed.")
            if output:
                st.code(output, language="text")

    st.divider()
    st.subheader("Manual Commands")
    st.code(
        "\n".join(
            [
                ".\\.venv\\Scripts\\python app/arxiv_loader.py",
                ".\\.venv\\Scripts\\python app/preprocessing.py",
                ".\\.venv\\Scripts\\python app/embedding_generation.py",
                ".\\.venv\\Scripts\\python app/vector_store.py",
                ".\\.venv\\Scripts\\python app/model_training.py",
                ".\\.venv\\Scripts\\python app/export_eda.py",
                ".\\.venv\\Scripts\\streamlit run app/app.py",
            ]
        ),
        language="bash",
    )
