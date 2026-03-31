# Model Summary

## Trained Models

### 1. Logistic Regression
- Type: Machine Learning model
- Pipeline: TF-IDF vectorization + Logistic Regression
- Saved artifact: `models/logistic_regression_pipeline.pkl`
- Accuracy: 0.785
- Macro Precision: 0.820
- Macro Recall: 0.662
- Macro F1: 0.710

### 2. MLP Classifier
- Type: ANN-style Deep Learning surrogate using a neural network classifier
- Pipeline: TF-IDF vectorization + MLPClassifier
- Saved artifact: `models/mlp_classifier_pipeline.pkl`
- Accuracy: 0.845
- Macro Precision: 0.800
- Macro Recall: 0.784
- Macro F1: 0.790

## Current Best Model

The current best local classification model is the `MLPClassifier`, which achieved the highest macro F1 score on the top-10-category arXiv classification task.

## Why These Models

- Logistic Regression gives a strong and interpretable ML baseline.
- MLPClassifier provides a neural-network-based comparison model that satisfies the fellowship requirement for a second, more advanced model.
- Both models are trained by us on the arXiv subset and saved locally for deployment and fallback use.
