# ResearchRAG AI

ResearchRAG AI is an arXiv-based academic assistant that combines Retrieval-Augmented Generation with two trained local classification models to satisfy the fellowship requirement for model development, comparison, and deployment.

## Problem Statement

Researchers and students often struggle to quickly identify the category and relevance of large numbers of academic papers. This project addresses that problem by combining:
- semantic retrieval over arXiv papers
- Gemini-based question answering with citations
- two trained local fallback models for paper-category prediction

## Fellowship Requirement Coverage

This project now includes:
- one ML model: TF-IDF + Logistic Regression
- one ANN-style model: TF-IDF + MLPClassifier
- saved trained artifacts in `models/`
- evaluation metrics in `reports/`
- Streamlit deployment-ready UI
- Gemini API integration for free-tier hosted answer generation
- local trained-model fallback mode when Gemini is unavailable

## Dataset Source

- arXiv metadata snapshot from Kaggle
- Expected file: `data/datasets/arxiv/arxiv-metadata-oai-snapshot.json`
- Kaggle link: https://www.kaggle.com/datasets/Cornell-University/arxiv

## Project Structure

```text
ResearchRAG-AI/
├── app/
│   ├── app.py
│   ├── arxiv_loader.py
│   ├── embedding_generation.py
│   ├── fallback_models.py
│   ├── generation.py
│   ├── model_training.py
│   ├── preprocessing.py
│   ├── retrieval.py
│   └── vector_store.py
├── data/
├── models/
├── notebooks/
├── reports/
├── .env.example
├── requirements.txt
└── README.md
```

## Environment Variables

Create `.env` from `.env.example` and set:

```env
GEMINI_API_KEY=your_google_gemini_key
GEMINI_MODEL=gemini-2.5-flash
ARXIV_MAX_RECORDS=5000
ARXIV_TOP_CATEGORIES=10
```

## Local Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Full Pipeline

```bash
.venv\Scripts\python app/arxiv_loader.py
.venv\Scripts\python app/preprocessing.py
.venv\Scripts\python app/embedding_generation.py
.venv\Scripts\python app/vector_store.py
.venv\Scripts\python app/model_training.py
.venv\Scripts\streamlit run app/app.py
```

## What The App Demonstrates

1. Research Q&A over retrieved arXiv context
2. Gemini-generated answers with citations when a key is available
3. Local trained fallback mode when Gemini is unavailable
4. Category prediction using two trained models
5. Model comparison metrics suitable for the report and presentation

## Saved Artifacts

- `models/logistic_regression_pipeline.pkl`
- `models/mlp_classifier_pipeline.pkl`
- `models/embeddings.npy`
- `models/faiss_index.bin`
- `reports/model_metrics.json`
- `reports/model_comparison.csv`

## Notes

- The local fallback mode is intentionally tied to trained models so the project is not dependent on a paid API.
- You should still create notebooks, a report PDF, slides, and a deployment link to fully complete the fellowship submission package.
