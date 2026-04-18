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
        grid-template-columns: 1.5fr 0.9fr;
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
        font-size: 3.2rem;
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
    .status-card {
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
        padding: 1.4rem;
        margin-top: 0.4rem;
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
        margin-bottom: 0.15rem;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
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
            <div class="hero-kicker">Your AI Research Companion</div>
            <div class="hero-title">ResearchRAG AI</div>
            <p class="hero-copy">
                Discover relevant papers faster, ask grounded research questions, and classify academic work
                with an interface designed for focused exploration.
            </p>
            <div class="hero-badges">
                <div class="hero-badge">Ask with citations</div>
                <div class="hero-badge">Search by meaning</div>
                <div class="hero-badge">Classify papers</div>
                <div class="hero-badge">Explore trends</div>
            </div>
        </div>
        <div class="hero-side-panel">
            <div class="hero-side-title">At a Glance</div>
            <div class="insight-stack">
                <div class="insight-item">
                    <div class="insight-label">Indexed Research Chunks</div>
                    <div class="insight-value">{vector_count}</div>
                </div>
                <div class="insight-item">
                    <div class="insight-label">Research Categories Covered</div>
                    <div class="insight-value">{top_categories}</div>
                </div>
                <div class="insight-item">
                    <div class="insight-label">Best Local Model F1</div>
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
    render_status_card("Search", "Ready" if index_ready else "Preparing", "Meaning-based retrieval", "warm")
with status_columns[1]:
    render_status_card("Answers", "Live" if gemini_ready else "Grounded", "Research Q&A experience", "cool")
with status_columns[2]:
    render_status_card("Classification", "Ready" if classification_ready else "Preparing", "Dual-model prediction", "earth")
with status_columns[3]:
    render_status_card("Experience", "Focused", "Built for end users", "rose")

feature_columns = st.columns(3)
with feature_columns[0]:
    render_feature_card(
        "Research questions with evidence",
        "Ask about methods, concepts, or themes and get answers supported by retrieved arXiv sources.",
    )
with feature_columns[1]:
    render_feature_card(
        "Smarter semantic discovery",
        "Find papers by meaning rather than relying only on exact title words or rigid keyword matches.",
    )
with feature_columns[2]:
    render_feature_card(
        "Quick paper categorization",
        "Paste a paper title and abstract to compare how both trained models classify the work.",
    )

qa_tab, search_tab, prediction_tab, evidence_tab = st.tabs(
    ["Ask", "Discover", "Classify", "Insights"]
)

with qa_tab:
    st.markdown("<div class='section-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Ask citation-grounded research questions</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Use natural language to explore topics, methods, findings, and scientific trends across the indexed research collection.</div>",
        unsafe_allow_html=True,
    )
    with st.form("qa_form"):
        question = st.text_area(
            "Your question",
            height=130,
            placeholder="Example: What themes appear in recent work on quantum error correction, and which retrieved papers support them?",
        )
        result_count = st.slider("How many sources should support the answer?", min_value=3, max_value=10, value=5, key="qa_slider")
        ask_clicked = st.form_submit_button("Generate Answer")

    if ask_clicked:
        if not question.strip():
            st.warning("Enter a research question first.")
        elif not index_ready:
            st.warning("Search resources are still preparing. Please try again shortly.")
        else:
            with st.spinner("Retrieving sources and preparing your answer..."):
                qa_results = retrieve_similar_chunks(question, top_k=result_count)
                answer, _mode = generate_answer(question, qa_results)
            st.markdown("### Answer")
            st.write(answer)
            render_context_results(qa_results, "Supporting Sources")

    st.markdown("</div>", unsafe_allow_html=True)

with search_tab:
    st.markdown("<div class='section-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Discover papers by meaning</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Search with a concept, technique, or problem statement to surface semantically related papers.</div>",
        unsafe_allow_html=True,
    )
    with st.form("semantic_search_form"):
        search_query = st.text_input(
            "Search topic",
            placeholder="Example: graph sparsity for large-scale optimization",
        )
        search_count = st.slider("How many results would you like?", min_value=3, max_value=10, value=5, key="search_slider")
        search_clicked = st.form_submit_button("Search Papers")

    if search_clicked:
        if not search_query.strip():
            st.warning("Enter a concept or topic to search.")
        elif not index_ready:
            st.warning("Search resources are still preparing. Please try again shortly.")
        else:
            with st.spinner("Searching for related papers..."):
                search_results = retrieve_similar_chunks(search_query, top_k=search_count)
            render_context_results(search_results, "Best Matches")

    st.markdown("</div>", unsafe_allow_html=True)

with prediction_tab:
    st.markdown("<div class='section-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Classify a paper</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Paste a title and abstract to see how the local classification models label the paper.</div>",
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
                st.warning("Classification resources are still preparing. Please try again shortly.")
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
                        <strong>Top prediction:</strong> {best_details["label"]} from {best_model_name}
                        at {best_details["confidence"]:.2%}.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("</div>", unsafe_allow_html=True)

with evidence_tab:
    st.markdown("<div class='section-panel'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Explore collection insights</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='section-copy'>Browse a quick snapshot of model quality and dataset patterns behind the experience.</div>",
        unsafe_allow_html=True,
    )

    evidence_cols = st.columns(3)
    with evidence_cols[0]:
        render_feature_card(
            "Collection size",
            f"The current research index contains {vector_count} searchable text chunks.",
        )
    with evidence_cols[1]:
        render_feature_card(
            "Strongest local model",
            f"The best saved local classifier currently reaches a macro F1 score of {best_model_value}.",
        )
    with evidence_cols[2]:
        render_feature_card(
            "Category coverage",
            f"The classification workflow is focused on the top {top_categories} research categories in the prepared subset.",
        )

    if metrics:
        st.markdown("### Model Quality")
        metrics_cols = st.columns(2)
        with metrics_cols[0]:
            logreg = metrics.get("logistic_regression", {})
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="evidence-title">Logistic Regression</div>
                    <div class="evidence-value">{logreg.get("f1_macro", 0.0):.3f}</div>
                    <div class="evidence-copy">Macro F1 score.</div>
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
                    <div class="evidence-copy">Macro F1 score.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    if comparison_df is not None:
        st.markdown("### Model Comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    if eda_summary:
        st.markdown("### Collection Summary")
        st.markdown(eda_summary)

    st.markdown("### Visual Trends")
    render_eda_images()
    st.markdown("</div>", unsafe_allow_html=True)

with st.sidebar:
    st.header("ResearchRAG AI")
    st.caption("Explore, discover, and understand research faster.")

    st.subheader("Try asking")
    st.markdown(
        "- What retrieval evidence supports current work on quantum optics?\n"
        "- Which papers relate to graph sparsity in optimization?\n"
        "- What themes appear across recent work on scientific language models?"
    )

    st.subheader("Try discovering")
    st.markdown(
        "- diffusion models for medical imaging\n"
        "- retrieval-augmented generation in science\n"
        "- topological phases in condensed matter"
    )

    st.subheader("Try classifying")
    st.caption("Paste a paper title and abstract to compare model predictions.")
