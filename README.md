# ResearchRAG AI

ResearchRAG AI is an arXiv-based academic assistant that combines paper classification, semantic search, and citation-grounded question answering in a Streamlit web app.

## Core Features

- Paper classification using two trained local models
  - TF-IDF + Logistic Regression
  - TF-IDF + MLP Classifier
- Semantic search over arXiv papers using sentence-transformer embeddings and FAISS
- Research question answering using Google Gemini API
- Local fallback mode when Gemini is unavailable
- Evaluation dashboard and EDA visualizations

## Dataset

- arXiv Metadata Snapshot (Kaggle)
- Link: https://www.kaggle.com/datasets/Cornell-University/arxiv

## Runtime Artifacts Used by the App

These files are required for the deployed app and are included in the repository:
- `models/faiss_index.bin`
- `models/embedding_metadata.jsonl`
- `models/faiss_index_metadata.json`
- `models/logistic_regression_pipeline.pkl`
- `models/mlp_classifier_pipeline.pkl`
- `models/label_encoder.pkl`
- `reports/eda_exports/*.png`

The raw arXiv dataset file is not included in GitHub because it is too large.

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

## Run The App Locally

```bash
.venv\Scripts\streamlit run app/app.py
```

## Streamlit Community Cloud Deployment

### Entrypoint
- File path: `app/app.py`

### Runtime dependencies
- Use `requirements.txt` for deployment

### Secrets
Add the following in Streamlit Community Cloud secrets:

```toml
GEMINI_API_KEY = "your_key_here"
GEMINI_MODEL = "gemini-2.5-flash"
ARXIV_MAX_RECORDS = "5000"
ARXIV_TOP_CATEGORIES = "10"
```

## Local Development Commands

```bash
.venv\Scripts\python app/arxiv_loader.py
.venv\Scripts\python app/preprocessing.py
.venv\Scripts\python app/embedding_generation.py
.venv\Scripts\python app/vector_store.py
.venv\Scripts\python app/model_training.py
.venv\Scripts\python app/export_eda.py
```

## Reports and Documentation

- `reports/model_metrics.json`
- `reports/model_comparison.csv`
- `reports/EDA_SUMMARY.md`
- `reports/SRS_ONE_PAGE_SUBMISSION.md`
- `notebooks/arxiv_eda.ipynb`

## Deployment Note

This repo is prepared so the app can run on Streamlit Community Cloud without the raw dataset file, because the necessary runtime model and FAISS artifacts are already included.
