import json
import os
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_TEXT_DIR = PROJECT_ROOT / "data/processed_text"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"
ARXIV_DOCUMENTS_PATH = PROCESSED_TEXT_DIR / "arxiv_documents.jsonl"
LOGREG_MODEL_PATH = MODELS_DIR / "logistic_regression_pipeline.pkl"
MLP_MODEL_PATH = MODELS_DIR / "mlp_classifier_pipeline.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
TOP_CATEGORIES = int(os.getenv("ARXIV_TOP_CATEGORIES", "10"))


def load_training_dataframe() -> pd.DataFrame:
    """Load normalized arXiv records and prepare a supervised dataset."""
    records = []
    with ARXIV_DOCUMENTS_PATH.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            record = json.loads(line)
            primary_category = (record.get("categories", "").split() or ["unknown"])[0]
            text = f"{record.get('title', '')} {record.get('abstract', '')}".strip()
            if text and primary_category != "unknown":
                records.append({"text": text, "label": primary_category})

    dataframe = pd.DataFrame(records)
    top_labels = [label for label, _ in Counter(dataframe["label"]).most_common(TOP_CATEGORIES)]
    return dataframe[dataframe["label"].isin(top_labels)].reset_index(drop=True)


def evaluate_model(model: Pipeline, x_test: pd.Series, y_test, label_encoder: LabelEncoder | None = None) -> dict:
    """Compute the evaluation metrics required for comparison."""
    predictions = model.predict(x_test)
    true_labels = y_test
    pred_labels = predictions

    if label_encoder is not None:
        true_labels = label_encoder.inverse_transform(y_test)
        pred_labels = label_encoder.inverse_transform(predictions)

    return {
        "accuracy": float(accuracy_score(true_labels, pred_labels)),
        "precision_macro": float(precision_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
        "classification_report": classification_report(true_labels, pred_labels, zero_division=0),
    }


def train_models() -> None:
    """Train, compare, and save one ML model and one ANN-style model."""
    if not ARXIV_DOCUMENTS_PATH.exists():
        print(f"Normalized arXiv file not found: {ARXIV_DOCUMENTS_PATH}")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = load_training_dataframe()

    x_train, x_test, y_train, y_test = train_test_split(
        dataframe["text"],
        dataframe["label"],
        test_size=0.2,
        random_state=42,
        stratify=dataframe["label"],
    )

    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    logreg_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=6000, ngram_range=(1, 2), stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )
    mlp_pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words="english")),
            ("classifier", MLPClassifier(hidden_layer_sizes=(64,), max_iter=20, early_stopping=False, random_state=42)),
        ]
    )

    logreg_pipeline.fit(x_train, y_train)
    mlp_pipeline.fit(x_train, y_train_encoded)

    logreg_metrics = evaluate_model(logreg_pipeline, x_test, y_test)
    mlp_metrics = evaluate_model(mlp_pipeline, x_test, y_test_encoded, label_encoder=label_encoder)

    joblib.dump(logreg_pipeline, LOGREG_MODEL_PATH, compress=3)
    joblib.dump(mlp_pipeline, MLP_MODEL_PATH, compress=3)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH, compress=3)

    metrics = {
        "dataset_size": int(len(dataframe)),
        "top_categories": int(TOP_CATEGORIES),
        "train_size": int(len(x_train)),
        "test_size": int(len(x_test)),
        "labels": label_encoder.classes_.tolist(),
        "logistic_regression": logreg_metrics,
        "mlp_classifier": mlp_metrics,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    comparison = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "accuracy": logreg_metrics["accuracy"],
                "precision_macro": logreg_metrics["precision_macro"],
                "recall_macro": logreg_metrics["recall_macro"],
                "f1_macro": logreg_metrics["f1_macro"],
            },
            {
                "model": "MLP Classifier",
                "accuracy": mlp_metrics["accuracy"],
                "precision_macro": mlp_metrics["precision_macro"],
                "recall_macro": mlp_metrics["recall_macro"],
                "f1_macro": mlp_metrics["f1_macro"],
            },
        ]
    )
    comparison.to_csv(COMPARISON_PATH, index=False)

    print(f"Trained models on {len(dataframe)} records across top {TOP_CATEGORIES} categories")
    print(f"Saved ML model to {LOGREG_MODEL_PATH}")
    print(f"Saved ANN-style model to {MLP_MODEL_PATH}")
    print(f"Saved label encoder to {LABEL_ENCODER_PATH}")
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    train_models()
