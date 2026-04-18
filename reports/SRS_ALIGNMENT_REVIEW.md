# SRS Alignment Review

## Overall Result

The current project aligns well with the stated SRS and now presents the required features in a cleaner deployment-facing interface.

## Requirement Check

| SRS Requirement | Status | Evidence |
| --- | --- | --- |
| Streamlit web app interface | Complete | `app/app.py` |
| Paper classification using ML models | Complete | Saved logistic regression and MLP pipelines in `models/` |
| Title + abstract to category prediction | Complete | UI now collects title and abstract separately, then combines them for prediction |
| Semantic search using sentence-transformers embeddings + FAISS | Complete | `app/retrieval.py`, `models/faiss_index.bin`, `models/embedding_metadata.jsonl` |
| Research Q&A with Gemini API | Complete | `app/generation.py` |
| Citation-grounded answers | Complete | Retrieval context includes arXiv citation metadata and is surfaced in the UI |
| Fallback mode when API is unavailable | Complete | `app/generation.py`, `app/fallback_models.py` |
| Evaluation metrics for models | Complete | `reports/model_metrics.json`, `reports/model_comparison.csv` |
| EDA visualizations and summary | Complete | `reports/EDA_SUMMARY.md`, `reports/eda_exports/` |
| Saved local artifacts included for deployment | Complete | `models/*.pkl`, FAISS artifacts committed in repo |
| GitHub repository ready for deployment | Complete | README, `.streamlit/config.toml`, and runtime assets are present |

## Improvements Made During Review

- Removed developer-facing pipeline buttons and manual shell commands from the deployed UI.
- Strengthened the app layout so the interface feels product-ready rather than like an internal tool.
- Aligned the classification workflow more explicitly with the SRS wording by separating title and abstract inputs.
- Improved retrieval performance by caching the embedding model, FAISS index, and metadata.
- Made model and retrieval paths more robust by resolving them from the project root.

## Remaining Notes

- Gemini-based answering still depends on a valid `GEMINI_API_KEY` in environment variables or Streamlit secrets.
- The raw arXiv dataset is intentionally excluded from GitHub and deployment, but the required runtime artifacts are already included.

