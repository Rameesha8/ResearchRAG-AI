import json
import os
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
FAISS_INDEX_PATH = PROJECT_ROOT / "models/faiss_index.bin"
FAISS_INFO_PATH = PROJECT_ROOT / "models/faiss_index_metadata.json"
METADATA_PATH = PROJECT_ROOT / "models/embedding_metadata.jsonl"
LOGREG_MODEL_PATH = PROJECT_ROOT / "models/logistic_regression_pipeline.pkl"
MLP_MODEL_PATH = PROJECT_ROOT / "models/mlp_classifier_pipeline.pkl"
METRICS_PATH = PROJECT_ROOT / "reports/model_metrics.json"
COMPARISON_PATH = PROJECT_ROOT / "reports/model_comparison.csv"
EDA_SUMMARY_PATH = PROJECT_ROOT / "reports/EDA_SUMMARY.md"
EDA_DIR = PROJECT_ROOT / "reports/eda_exports"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

st.set_page_config(page_title="ResearchRAG AI", page_icon=":material/auto_stories:", layout="wide")


@st.cache_data(show_spinner=False)
def load_json_file(path: Path) -> dict | None:
    """Load JSON content when available."""
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def load_comparison_table() -> pd.DataFrame | None:
    """Load saved model comparison metrics if present."""
    if not COMPARISON_PATH.exists():
        return None
    return pd.read_csv(COMPARISON_PATH)


@st.cache_data(show_spinner=False)
def load_markdown_file(path: Path) -> str | None:
    """Load a markdown report if present."""
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")


def render_status_card(title: str, value: str, caption: str, tone: str) -> None:
    """Render a compact status card."""
    st.markdown(
        f"""
        <div class="status-card {tone}">
            <div class="status-label">{title}</div>
            <div class="status-value">{value}</div>
            <div class="status-caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(title: str, body: str) -> None:
    """Render a concise feature card."""
    st.markdown(
        f"""
        <div class="feature-card">
            <div class="feature-title">{title}</div>
            <div class="feature-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_card(model_name: str, label: str, confidence: float, accent_class: str) -> None:
    """Render a prediction result card."""
    st.markdown(
        f"""
        <div class="prediction-card {accent_class}">
            <div class="prediction-model">{model_name}</div>
            <div class="prediction-label">{label}</div>
            <div class="prediction-confidence">Confidence: {confidence:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_context_results(results: list[dict], title: str) -> None:
    """Render retrieved sources with citation-ready metadata."""
    st.markdown(f"### {title}")
    if not results:
        st.info("No supporting sources were retrieved.")
        return

    for index, result in enumerate(results, start=1):
        title_text = result.get("title", "Untitled")
        citation = result.get("document_id", "Unknown")
        meta_line = (
            f"{result.get('authors', 'Unknown authors')}  |  "
            f"{result.get('categories', 'Unknown categories')}  |  "
            f"Updated {result.get('update_date', 'Unknown date')}"
        )
        with st.container():
            st.markdown(
                f"""
                <div class="source-card">
                    <div class="source-rank">Source {index}</div>
                    <div class="source-title">{title_text}</div>
                    <div class="source-meta">arXiv:{citation}  |  {meta_line}</div>
                    <div class="source-score">Vector distance: {result.get('score', 0.0):.4f}</div>
                </div>
                """,
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


metrics = load_json_file(METRICS_PATH) or {}
comparison_df = load_comparison_table()
faiss_info = load_json_file(FAISS_INFO_PATH) or {}
eda_summary = load_markdown_file(EDA_SUMMARY_PATH)

dataset_ready = ARXIV_DATASET_PATH.exists()
index_ready = FAISS_INDEX_PATH.exists() and METADATA_PATH.exists()
classification_ready = LOGREG_MODEL_PATH.exists() and MLP_MODEL_PATH.exists()
gemini_ready = bool(os.getenv("GEMINI_API_KEY"))
vector_count = faiss_info.get("vector_count", "Unknown")
top_categories = metrics.get("top_categories", "Unknown")
best_model_f1 = metrics.get("mlp_classifier", {}).get("f1_macro")
best_model_value = f"{best_model_f1:.3f}" if isinstance(best_model_f1, (int, float)) else "Unavailable"

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Sans+3:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: "Source Sans 3", sans-serif;
    }
    .stApp {
        color: #172031;
        background:
            radial-gradient(circle at 12% 12%, rgba(245, 174, 74, 0.26), transparent 24%),
            radial-gradient(circle at 88% 10%, rgba(36, 143, 155, 0.22), transparent 25%),
            radial-gradient(circle at 50% 88%, rgba(17, 76, 95, 0.12), transparent 24%),
            linear-gradient(180deg, #f5efe3 0%, #f8f7f2 46%, #eef4f7 100%);
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.6rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, .hero-title, .section-title, .feature-title, .prediction-label, .source-title {
        font-family: "Space Grotesk", sans-serif;
    }
    .hero-grid {
        display: grid;
        grid-template-columns: 1.45fr 0.95fr;
        gap: 1rem;
        margin-bottom: 1.2rem;
    }
    .hero-panel, .hero-side-panel {
        position: relative;
        overflow: hidden;
        border-radius: 30px;
        padding: 2rem;
        border: 1px solid rgba(17, 42, 58, 0.08);
        box-shadow: 0 25px 70px rgba(18, 35, 45, 0.12);
    }
    .hero-panel {
        color: #f9fbfd;
        background:
            linear-gradient(135deg, rgba(14, 55, 67, 0.94) 0%, rgba(23, 96, 111, 0.92) 48%, rgba(239, 164, 78, 0.88) 120%);
    }
    .hero-panel::after {
        content: "";
        position: absolute;
        inset: auto -18% -28% auto;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(255,255,255,0.25), transparent 65%);
        filter: blur(8px);
    }
    .hero-side-panel {
        background: rgba(255, 255, 255, 0.72);
        backdrop-filter: blur(14px);
    }
    .hero-kicker {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.18em;
        opacity: 0.78;
        margin-bottom: 0.75rem;
    }
    .hero-title {
        font-size: 3.1rem;
        line-height: 0.98;
        font-weight: 700;
        margin: 0 0 0.9rem 0;
        max-width: 10ch;
    }
    .hero-copy {
        font-size: 1.04rem;
        line-height: 1.68;
        max-width: 64ch;
        margin: 0;
        color: rgba(249, 251, 253, 0.94);
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
        margin-top: 1.25rem;
    }
    .hero-badge {
        padding: 0.55rem 0.92rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.18);
        background: rgba(255, 255, 255, 0.12);
        font-size: 0.88rem;
    }
    .hero-side-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        color: #113847;
    }
    .insight-stack {
        display: grid;
        gap: 0.75rem;
    }
    .insight-item {
        padding: 0.95rem 1rem;
        border-radius: 20px;
        background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(244,248,250,0.8));
        border: 1px solid rgba(17, 42, 58, 0.08);
    }
    .insight-label {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #5c6d78;
        margin-bottom: 0.3rem;
    }
    .insight-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #173342;
    }
    .status-card {
        border-radius: 24px;
        padding: 1rem;
        min-height: 118px;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.45);
    }
    .status-card.warm {
        background: linear-gradient(180deg, #fff2d7 0%, #ffe1aa 100%);
    }
    .status-card.cool {
        background: linear-gradient(180deg, #ddf3ff 0%, #c9e5ff 100%);
    }
    .status-card.earth {
        background: linear-gradient(180deg, #e4f4e6 0%, #d2e8d6 100%);
    }
    .status-card.rose {
        background: linear-gradient(180deg, #fde8e1 0%, #f8d1c5 100%);
    }
    .status-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: rgba(23, 35, 49, 0.72);
        margin-bottom: 0.5rem;
    }
    .status-value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-bottom: 0.22rem;
    }
    .status-caption {
        color: #495869;
        font-size: 0.94rem;
    }
    .mode-banner {
        margin: 1rem 0 1.2rem 0;
        border-radius: 18px;
        padding: 1rem 1.1rem;
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(17, 42, 58, 0.08);
        box-shadow: 0 12px 30px rgba(18, 35, 45, 0.06);
    }
    .feature-card, .section-panel, .prediction-card, .source-card, .evidence-card {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid rgba(17, 42, 58, 0.08);
        box-shadow: 0 14px 40px rgba(18, 35, 45, 0.08);
    }
    .feature-card {
        border-radius: 22px;
        padding: 1rem;
        min-height: 150px;
    }
    .feature-title {
        font-size: 1.08rem;
        font-weight: 700;
        color: #123645;
        margin-bottom: 0.45rem;
    }
    .feature-copy {
        color: #4f5e6f;
        line-height: 1.55;
    }
    .section-panel {
        border-radius: 28px;
        padding: 1.2rem 1.2rem 1.4rem 1.2rem;
    }
    .section-title {
        font-size: 1.4rem;
        color: #123847;
        margin-bottom: 0.25rem;
    }
    .section-copy {
        color: #556476;
        margin-bottom: 1rem;
    }
    .prediction-card {
        border-radius: 24px;
        padding: 1rem;
        min-height: 132px;
    }
    .prediction-card.teal {
        background: linear-gradient(180deg, rgba(225, 246, 247, 0.98), rgba(242, 251, 251, 0.92));
    }
    .prediction-card.gold {
        background: linear-gradient(180deg, rgba(255, 242, 220, 0.98), rgba(252, 248, 240, 0.92));
    }
    .prediction-model {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #5b6976;
    }
    .prediction-label {
        font-size: 1.45rem;
        font-weight: 700;
        margin: 0.45rem 0 0.2rem 0;
        color: #123645;
    }
    .prediction-confidence {
        color: #526173;
        font-size: 0.95rem;
    }
    .consensus-banner {
        margin-top: 1rem;
        padding: 0.95rem 1rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(18, 68, 85, 0.95), rgba(31, 118, 133, 0.92));
        color: #f7fbfd;
    }
    .source-card {
        border-radius: 22px;
        padding: 0.95rem 1rem 0.8rem 1rem;
        margin: 0.9rem 0 0.55rem 0;
    }
    .source-rank {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #60717d;
        margin-bottom: 0.28rem;
    }
    .source-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #143949;
        margin-bottom: 0.18rem;
    }
    .source-meta, .source-score {
        color: #566777;
        font-size: 0.92rem;
    }
    .source-score {
        margin-top: 0.3rem;
    }
    .evidence-card {
        border-radius: 22px;
        padding: 1rem;
        min-height: 120px;
    }
    .evidence-title {
        font-size: 1rem;
        font-weight: 700;
        color: #143949;
        margin-bottom: 0.35rem;
    }
    .evidence-value {
        font-family: "Space Grotesk", sans-serif;
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 0.18rem;
        color: #143949;
    }
    .evidence-copy {
        color: #5b6976;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
        margin-bottom: 0.9rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        background: rgba(255,255,255,0.74);
        border: 1px solid rgba(17, 42, 58, 0.08);
        padding: 0.65rem 1rem;
        height: auto;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #104754 0%, #1d6673 100%);
        color: white;
    }
    .stTextInput input, .stTextArea textarea {
        border-radius: 18px !important;
        border: 1px solid rgba(17, 42, 58, 0.12) !important;
        background: rgba(255, 255, 255, 0.94) !important;
    }
    .stButton > button, .stForm button {
        border-radius: 999px;
        border: none;
        background: linear-gradient(135deg, #104754 0%, #1e7483 100%);
        color: white;
        font-weight: 700;
        padding: 0.6rem 1.05rem;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f1ece1 0%, #eef4f7 100%);
        border-right: 1px solid rgba(17, 42, 58, 0.08);
    }
    @media (max-width: 900px) {
        .hero-grid {
            grid-template-columns: 1fr;
        }
        .hero-title {
            font-size: 2.35rem;
            max-width: none;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="hero-grid">
        <div class="hero-panel">
            <div class="hero-kicker">Citation-grounded academic assistant</div>
            <div class="hero-title">ResearchRAG AI</div>
            <p class="hero-copy">
                Search the arXiv subset semantically, classify papers from title plus abstract,
                and answer research questions with retrieval-backed evidence. The interface is
                tuned for exploration, not pipeline debugging.
            </p>
            <div class="hero-badges">
                <div class="hero-badge">Semantic Retrieval</div>
                <div class="hero-badge">Gemini + Fallback</div>
                <div class="hero-badge">Dual Classifiers</div>
                <div class="hero-badge">Evaluation Ready</div>
            </div>
        </div>
        <div class="hero-side-panel">
            <div class="hero-side-title">Project Snapshot</div>
            <div class="insight-stack">
                <div class="insight-item">
                    <div class="insight-label">Indexed Chunks</div>
                    <div class="insight-value">{vector_count}</div>
                </div>
                <div class="insight-item">
                    <div class="insight-label">Top Categories Used for Classification</div>
                    <div class="insight-value">{top_categories}</div>
                </div>
                <div class="insight-item">
                    <div class="insight-label">Best Local Macro F1</div>
                    <div class="insight-value">{best_model_value}</div>
                </div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

status_columns = st.columns(4)
with status_columns[0]:
    render_status_card("Dataset", "Ready" if dataset_ready else "Missing", "Raw corpus available locally", "warm")
with status_columns[1]:
    render_status_card("Semantic Index", "Ready" if index_ready else "Pending", "FAISS retrieval assets", "cool")
with status_columns[2]:
    render_status_card("Classification", "Ready" if classification_ready else "Pending", "Dual local models", "earth")
with status_columns[3]:
    render_status_card("Generation", "Gemini" if gemini_ready else "Fallback", "Live API or local evidence mode", "rose")

if gemini_ready:
    st.markdown(
        f"<div class='mode-banner'><strong>Live answer generation is enabled.</strong> Gemini is configured with <code>{GEMINI_MODEL}</code>, and the app will still fall back to retrieval-grounded local mode if the API fails.</div>",
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        "<div class='mode-banner'><strong>Fallback mode is active.</strong> The app still supports semantic retrieval and local category prediction, and research answers are composed from retrieved evidence when Gemini is unavailable.</div>",
        unsafe_allow_html=True,
    )

feature_columns = st.columns(3)
with feature_columns[0]:
    render_feature_card(
        "Ask better research questions",
        "Use natural language instead of exact keyword matching. The system retrieves semantically relevant arXiv records before answering.",
    )
with feature_columns[1]:
    render_feature_card(
        "Classify papers the SRS way",
        "The classifier now accepts a paper title and abstract separately, matching the stated requirement for category prediction.",
    )
with feature_columns[2]:
    render_feature_card(
        "Keep the product surface clean",
        "Developer pipeline controls and manual shell commands have been removed from the main UI so the app feels deployment-ready.",
    )

qa_tab, search_tab, prediction_tab, evidence_tab = st.tabs(
    ["Research Q&A", "Semantic Search", "Category Prediction", "Project Evidence"]
)

with qa_tab:
    st.markdown("<div class='section-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Citation-grounded answers</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Ask about methods, topics, trends, or technical concepts. Answers are generated only after retrieving supporting arXiv context.</div>",
        unsafe_allow_html=True,
    )
    with st.form("qa_form"):
        question = st.text_area(
            "Research question",
            height=130,
            placeholder="Example: What themes appear in recent work on quantum error correction, and which retrieved papers support them?",
        )
        result_count = st.slider("Sources to use", min_value=3, max_value=10, value=5, key="qa_slider")
        ask_clicked = st.form_submit_button("Generate Answer")

    if ask_clicked:
        if not question.strip():
            st.warning("Enter a research question first.")
        elif not index_ready:
            st.warning("The semantic index is not available yet, so retrieval-backed Q&A cannot run.")
        else:
            with st.spinner("Retrieving supporting papers and composing the answer..."):
                qa_results = retrieve_similar_chunks(question, top_k=result_count)
                answer, mode = generate_answer(question, qa_results)
            st.markdown("### Answer")
            st.write(answer)
            st.caption(f"Answer mode: {mode}")
            render_context_results(qa_results, "Supporting Sources")

    st.markdown("</div>", unsafe_allow_html=True)

with search_tab:
    st.markdown("<div class='section-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Meaning-based paper search</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Search by concept, method, or problem statement. This is useful when exact titles or keywords are unknown.</div>",
        unsafe_allow_html=True,
    )
    with st.form("semantic_search_form"):
        search_query = st.text_input(
            "Semantic search query",
            placeholder="Example: graph sparsity for large-scale optimization",
        )
        search_count = st.slider("Results to show", min_value=3, max_value=10, value=5, key="search_slider")
        search_clicked = st.form_submit_button("Run Semantic Search")

    if search_clicked:
        if not search_query.strip():
            st.warning("Enter a concept or topic to search.")
        elif not index_ready:
            st.warning("The semantic index is not available yet.")
        else:
            with st.spinner("Searching the vector index..."):
                search_results = retrieve_similar_chunks(search_query, top_k=search_count)
            render_context_results(search_results, "Best Matching Papers")

    st.markdown("</div>", unsafe_allow_html=True)

with prediction_tab:
    st.markdown("<div class='section-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Paper category prediction</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Paste the title and abstract of a paper. Both trained local models score the combined text so you can compare their predictions side by side.</div>",
        unsafe_allow_html=True,
    )
    with st.form("prediction_form"):
        paper_title = st.text_input(
            "Paper title",
            placeholder="Example: Learning sparse graph representations for molecular reasoning",
        )
        paper_abstract = st.text_area(
            "Paper abstract",
            height=220,
            placeholder="Paste the abstract here...",
        )
        predict_clicked = st.form_submit_button("Predict Category")

    if predict_clicked:
        combined_text = "\n\n".join(part.strip() for part in [paper_title, paper_abstract] if part.strip())
        if not combined_text:
            st.warning("Add a paper title, an abstract, or both before running prediction.")
        else:
            predictions = predict_with_local_models(combined_text)
            if not predictions:
                st.warning("The trained local models are not available yet.")
            else:
                prediction_cols = st.columns(2)
                if "logistic_regression" in predictions:
                    details = predictions["logistic_regression"]
                    with prediction_cols[0]:
                        render_prediction_card(
                            "Logistic Regression",
                            details["label"],
                            details["confidence"],
                            "gold",
                        )
                if "mlp_classifier" in predictions:
                    details = predictions["mlp_classifier"]
                    with prediction_cols[1]:
                        render_prediction_card(
                            "MLP Classifier",
                            details["label"],
                            details["confidence"],
                            "teal",
                        )

                best_prediction = max(predictions.items(), key=lambda item: item[1]["confidence"])
                best_model_name = best_prediction[0].replace("_", " ").title()
                best_details = best_prediction[1]
                st.markdown(
                    f"""
                    <div class="consensus-banner">
                        <strong>Highest-confidence prediction:</strong> {best_details["label"]} from {best_model_name}
                        at {best_details["confidence"]:.2%}.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

with evidence_tab:
    st.markdown("<div class='section-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Evaluation and dataset evidence</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>This section keeps the most important supporting evidence visible for reviewers, recruiters, or deployment judges.</div>",
        unsafe_allow_html=True,
    )

    evidence_cols = st.columns(3)
    with evidence_cols[0]:
        render_feature_card(
            "Dataset footprint",
            f"Indexed vectors: {vector_count}. Local raw dataset status: {'available' if dataset_ready else 'not bundled in deployment build'}.",
        )
    with evidence_cols[1]:
        render_feature_card(
            "Best local model",
            f"MLP classifier macro F1: {best_model_value}. This is the strongest local fallback classifier in the saved evaluation report.",
        )
    with evidence_cols[2]:
        render_feature_card(
            "Fallback resilience",
            "If Gemini is unavailable, the app still answers using retrieval evidence and still supports both local classifiers.",
        )

    if metrics:
        st.markdown("### Model Metrics")
        metrics_cols = st.columns(2)
        with metrics_cols[0]:
            logreg = metrics.get("logistic_regression", {})
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-title">Logistic Regression</div>
                    <div class="evidence-value">{logreg.get("f1_macro", 0.0):.3f}</div>
                    <div class="evidence-copy">Macro F1 with TF-IDF features.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with metrics_cols[1]:
            mlp = metrics.get("mlp_classifier", {})
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-title">MLP Classifier</div>
                    <div class="evidence-value">{mlp.get("f1_macro", 0.0):.3f}</div>
                    <div class="evidence-copy">Best-performing saved local classifier.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if comparison_df is not None:
        st.markdown("### Model Comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    if eda_summary:
        st.markdown("### EDA Summary")
        st.markdown(eda_summary)

    st.markdown("### EDA Visuals")
    render_eda_images()
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("ResearchRAG AI")
    st.caption("A cleaner deployment-facing control panel.")

    st.subheader("Availability")
    st.write(f"{'Ready' if dataset_ready else 'Missing'} Dataset")
    st.write(f"{'Ready' if index_ready else 'Pending'} Semantic index")
    st.write(f"{'Ready' if classification_ready else 'Pending'} Classification models")
    st.write(f"{'Gemini' if gemini_ready else 'Fallback'} Answer mode")

    st.subheader("Recommended prompts")
    st.markdown(
        "- What retrieval evidence supports current work on quantum optics?\n"
        "- Find papers related to graph sparsity in optimization.\n"
        "- Predict the likely arXiv category for this abstract."
    )

    st.subheader("Deployment notes")
    st.caption(
        "Use `app/app.py` as the Streamlit entrypoint. Configure `GEMINI_API_KEY` in Streamlit secrets to enable live generation."
    )
