import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
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
COMPARISON_PATH = PROJECT_ROOT / "reports/model_comparison.csv"
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


def load_comparison_table() -> pd.DataFrame | None:
    """Load saved model comparison table if present."""
    if not COMPARISON_PATH.exists():
        return None
    return pd.read_csv(COMPARISON_PATH)


def render_context_results(results: list[dict], title: str = "Supporting Sources") -> None:
    """Render retrieved context cleanly for presentation."""
    st.markdown(f"### {title}")
    if not results:
        st.info("No retrieval results were found.")
        return

    for index, result in enumerate(results, start=1):
        title_text = result.get("title", "Untitled")
        with st.expander(f"{index}. {title_text}", expanded=index == 1):
            st.caption(
                f"arXiv:{result.get('document_id', 'Unknown')}  |  "
                f"{result.get('authors', 'Unknown authors')}  |  "
                f"{result.get('categories', 'Unknown categories')}  |  "
                f"Updated {result.get('update_date', 'Unknown date')}"
            )
            st.markdown(
                f"<div class='score-chip'>Similarity score: {result.get('score', 0.0):.4f}</div>",
                unsafe_allow_html=True,
            )
            st.write(result.get("text", ""))


def render_eda_images() -> None:
    """Render exported EDA charts if they exist."""
    plots = [
        ("Top Category Distribution", EDA_DIR / "top_category_distribution.png"),
        ("Title Length Distribution", EDA_DIR / "title_length_distribution.png"),
        ("Abstract Length Distribution", EDA_DIR / "abstract_length_distribution.png"),
    ]
    for heading, image_path in plots:
        if image_path.exists():
            st.markdown(f"**{heading}**")
            st.image(str(image_path), use_container_width=True)
        else:
            st.info(f"EDA asset not found: {image_path.name}")


def render_status_card(title: str, value: str, tone: str) -> None:
    """Render a custom metric-like card."""
    st.markdown(
        f"""
        <div class="status-card {tone}">
            <div class="status-label">{title}</div>
            <div class="status-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="ResearchRAG AI", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(255, 189, 89, 0.22), transparent 30%),
            radial-gradient(circle at top right, rgba(14, 165, 233, 0.18), transparent 28%),
            linear-gradient(180deg, #f6efe3 0%, #fcfaf6 46%, #f8fbfd 100%);
        color: #18212c;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }
    h1, h2, h3 {
        color: #132238;
        letter-spacing: -0.02em;
    }
    .hero-shell {
        padding: 2rem 2.2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, #0f3b48 0%, #145c66 42%, #f0a045 115%);
        box-shadow: 0 24px 60px rgba(15, 59, 72, 0.20);
        color: #f8fbfd;
        overflow: hidden;
        margin-bottom: 1.4rem;
    }
    .hero-kicker {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        opacity: 0.85;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-size: 3rem;
        line-height: 1.02;
        font-weight: 800;
        margin: 0 0 0.75rem 0;
        max-width: 10ch;
    }
    .hero-copy {
        font-size: 1.02rem;
        line-height: 1.65;
        max-width: 62ch;
        margin: 0;
        color: rgba(248, 251, 253, 0.92);
    }
    .hero-badge-row {
        display: flex;
        gap: 0.7rem;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }
    .hero-badge {
        border-radius: 999px;
        padding: 0.55rem 0.9rem;
        font-size: 0.88rem;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.14);
        backdrop-filter: blur(10px);
    }
    .section-panel {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(19, 34, 56, 0.08);
        border-radius: 24px;
        padding: 1.1rem 1.2rem 1.2rem 1.2rem;
        box-shadow: 0 14px 40px rgba(18, 34, 56, 0.08);
    }
    .status-card {
        border-radius: 22px;
        padding: 1rem 1rem 1.05rem 1rem;
        min-height: 108px;
        color: #0f1720;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.45);
    }
    .status-card.warm {
        background: linear-gradient(180deg, #fff2d6 0%, #ffe4b2 100%);
    }
    .status-card.cool {
        background: linear-gradient(180deg, #def4ff 0%, #cde8ff 100%);
    }
    .status-card.earth {
        background: linear-gradient(180deg, #e6f4ea 0%, #d6ecdb 100%);
    }
    .status-card.rose {
        background: linear-gradient(180deg, #fce6de 0%, #f9d5ca 100%);
    }
    .status-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.7;
        margin-bottom: 0.55rem;
    }
    .status-value {
        font-size: 1.55rem;
        font-weight: 750;
        line-height: 1.1;
    }
    .mode-banner {
        border-radius: 18px;
        padding: 0.95rem 1rem;
        margin: 0.35rem 0 1rem 0;
        border: 1px solid rgba(19, 34, 56, 0.08);
        background: rgba(255, 255, 255, 0.72);
        font-size: 0.98rem;
    }
    .score-chip {
        display: inline-block;
        border-radius: 999px;
        padding: 0.28rem 0.68rem;
        margin-bottom: 0.8rem;
        background: #e6f1f4;
        color: #17404d;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .prediction-card {
        border-radius: 22px;
        padding: 1rem 1rem 0.9rem 1rem;
        background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(244, 248, 250, 0.88));
        border: 1px solid rgba(19, 34, 56, 0.08);
        box-shadow: 0 12px 24px rgba(19, 34, 56, 0.06);
    }
    .prediction-model {
        font-size: 0.88rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 0.68;
    }
    .prediction-label {
        font-size: 1.45rem;
        font-weight: 760;
        margin: 0.5rem 0 0.2rem 0;
        color: #12303b;
    }
    .prediction-confidence {
        color: #38556b;
        font-size: 0.95rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
        margin-bottom: 0.8rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        background: rgba(255,255,255,0.74);
        border: 1px solid rgba(19, 34, 56, 0.08);
        padding: 0.65rem 1rem;
        height: auto;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(180deg, #144854 0%, #1d6673 100%);
        color: white;
    }
    .stTextInput input, .stTextArea textarea {
        border-radius: 16px !important;
        border: 1px solid rgba(19, 34, 56, 0.10) !important;
        background: rgba(255, 255, 255, 0.92) !important;
    }
    .stButton > button {
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #104754 0%, #1f7685 100%);
        color: white;
        font-weight: 650;
        padding: 0.6rem 1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #0d3c46 0%, #1a6672 100%);
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f2efe7 0%, #eef4f7 100%);
        border-right: 1px solid rgba(19, 34, 56, 0.08);
    }
    @media (max-width: 900px) {
        .hero-title {
            font-size: 2.3rem;
            max-width: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-shell">
        <div class="hero-kicker">Intelligent Academic Research Assistant</div>
        <div class="hero-title">ResearchRAG AI</div>
        <p class="hero-copy">
            Explore the arXiv corpus with a sharper interface for semantic search, topic prediction,
            and citation-grounded research answers. Built to feel like a polished assistant, not a
            debugging console.
        </p>
        <div class="hero-badge-row">
            <div class="hero-badge">Semantic Search</div>
            <div class="hero-badge">Gemini Answers</div>
            <div class="hero-badge">Local Fallback</div>
            <div class="hero-badge">Dual Model Classification</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

status_cols = st.columns(4)
with status_cols[0]:
    render_status_card("Dataset", "Ready" if ARXIV_DATASET_PATH.exists() else "Missing", "warm")
with status_cols[1]:
    render_status_card(
        "Semantic Index",
        "Ready" if FAISS_INDEX_PATH.exists() and METADATA_PATH.exists() else "Pending",
        "cool",
    )
with status_cols[2]:
    render_status_card("ML Model", "Ready" if LOGREG_MODEL_PATH.exists() else "Pending", "earth")
with status_cols[3]:
    render_status_card("DL Model", "Ready" if MLP_MODEL_PATH.exists() else "Pending", "rose")

if os.getenv("GEMINI_API_KEY"):
    st.markdown(
        f"<div class='mode-banner'><strong>Live Generation Enabled.</strong> Gemini is connected with model <code>{GEMINI_MODEL}</code>.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='mode-banner'><strong>Fallback Mode Active.</strong> Gemini is not configured, so answers will use trained local models plus retrieval evidence.</div>",
        unsafe_allow_html=True,
    )

main_tab1, main_tab2, main_tab3, main_tab4 = st.tabs(
    ["Research Q&A", "Semantic Search", "Category Prediction", "Evaluation & EDA"]
)

with main_tab1:
    st.markdown("<div class='section-panel'>", unsafe_allow_html=True)
    st.subheader("Ask a Research Question")
    question = st.text_input("Type a question about the arXiv corpus")
    result_count = st.slider("Sources to use", min_value=3, max_value=10, value=5, key="qa_slider")

    if question:
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            st.warning("The semantic index is not ready yet.")
        else:
            with st.spinner("Retrieving relevant papers and composing an answer..."):
                results = retrieve_similar_chunks(question, top_k=result_count)
                answer, mode = generate_answer(question, results)

            st.markdown("### Answer")
            st.write(answer)
            st.caption(f"Answer mode: {mode}")
            render_context_results(results)
    else:
        st.caption("Try asking about a topic, method, category, or scientific concept.")
    st.markdown("</div>", unsafe_allow_html=True)

with main_tab2:
    st.markdown("<div class='section-panel'>", unsafe_allow_html=True)
    st.subheader("Semantic Search")
    search_query = st.text_input("Search by meaning instead of exact keywords", key="semantic_query")
    search_count = st.slider("Results to show", min_value=3, max_value=10, value=5, key="search_slider")

    if search_query:
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            st.warning("The semantic index is not ready yet.")
        else:
            with st.spinner("Searching the vector index..."):
                search_results = retrieve_similar_chunks(search_query, top_k=search_count)
            render_context_results(search_results, title="Best Matching Papers")
    else:
        st.caption("Use a concept like 'quantum optics', 'graph sparsity', or 'diphoton production'.")
    st.markdown("</div>", unsafe_allow_html=True)

with main_tab3:
    st.markdown("<div class='section-panel'>", unsafe_allow_html=True)
    st.subheader("Predict Paper Category")
    paper_text = st.text_area("Paste a paper title and/or abstract", height=220)
    if st.button("Predict Category") and paper_text.strip():
        local_predictions = predict_with_local_models(paper_text)
        if local_predictions:
            pred_col1, pred_col2 = st.columns(2)
            if "logistic_regression" in local_predictions:
                details = local_predictions["logistic_regression"]
                pred_col1.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="prediction-model">Logistic Regression</div>
                        <div class="prediction-label">{details['label']}</div>
                        <div class="prediction-confidence">Confidence: {details['confidence']:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            if "mlp_classifier" in local_predictions:
                details = local_predictions["mlp_classifier"]
                pred_col2.markdown(
                    f"""
                    <div class="prediction-card">
                        <div class="prediction-model">MLP Classifier</div>
                        <div class="prediction-label">{details['label']}</div>
                        <div class="prediction-confidence">Confidence: {details['confidence']:.2%}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.warning("Trained local models are not ready yet.")
    st.markdown("</div>", unsafe_allow_html=True)

with main_tab4:
    st.markdown("<div class='section-panel'>", unsafe_allow_html=True)
    st.subheader("Model Evaluation")
    metrics = load_metrics()
    comparison_df = load_comparison_table()

    if metrics:
        metric_col1, metric_col2 = st.columns(2)
        metric_col1.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-model">Logistic Regression</div>
                <div class="prediction-label">{metrics['logistic_regression']['f1_macro']:.3f}</div>
                <div class="prediction-confidence">Macro F1</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        metric_col2.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-model">MLP Classifier</div>
                <div class="prediction-label">{metrics['mlp_classifier']['f1_macro']:.3f}</div>
                <div class="prediction-confidence">Macro F1</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.info("Training metrics are not available yet.")

    if comparison_df is not None:
        st.markdown("### Model Comparison Table")
        st.dataframe(comparison_df, use_container_width=True)

    st.markdown("### EDA Visuals")
    render_eda_images()
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Control Room")
    st.caption("Keep the main page clean. Use this area for setup and refresh tasks.")

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

    with st.expander("Run Pipeline Tasks", expanded=False):
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

    with st.expander("Manual Commands", expanded=False):
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
