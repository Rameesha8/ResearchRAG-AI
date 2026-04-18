from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
LOGREG_MODEL_PATH = MODELS_DIR / "logistic_regression_pipeline.pkl"
MLP_MODEL_PATH = MODELS_DIR / "mlp_classifier_pipeline.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"


@lru_cache(maxsize=1)
def load_models() -> tuple[object | None, object | None, object | None]:
    """Load trained fallback classifiers if they exist."""
    logreg_model = joblib.load(LOGREG_MODEL_PATH) if LOGREG_MODEL_PATH.exists() else None
    mlp_model = joblib.load(MLP_MODEL_PATH) if MLP_MODEL_PATH.exists() else None
    label_encoder = joblib.load(LABEL_ENCODER_PATH) if LABEL_ENCODER_PATH.exists() else None
    return logreg_model, mlp_model, label_encoder


def _predict_single(model: object, text: str, label_encoder: object | None = None) -> dict:
    """Generate a label and confidence from a scikit-learn text pipeline."""
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_
    best_index = int(np.argmax(probabilities))
    label = classes[best_index]
    if label_encoder is not None and isinstance(label, (int, np.integer)):
        label = label_encoder.inverse_transform([int(label)])[0]
    return {
        "label": str(label),
        "confidence": float(probabilities[best_index]),
    }


def predict_with_local_models(text: str) -> dict:
    """Run both trained local models on a piece of text."""
    logreg_model, mlp_model, label_encoder = load_models()
    predictions = {}

    if logreg_model is not None:
        predictions["logistic_regression"] = _predict_single(logreg_model, text)
    if mlp_model is not None:
        predictions["mlp_classifier"] = _predict_single(mlp_model, text, label_encoder=label_encoder)

    return predictions
