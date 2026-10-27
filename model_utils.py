"""Shared constants and evaluation helpers used by both training and the Streamlit app."""

from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "model"
RAW_DATA_PATH = PROJECT_ROOT / "data" / "students_dropout_academic_success.csv"
TEST_DATA_PATH = PROJECT_ROOT / "test_data.csv"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
METRICS_PATH = MODEL_DIR / "metrics_holdout.json"

TARGET_COLUMN = "Target"
RANDOM_STATE = 42

# Display name -> file stem inside model/
MODEL_FILES = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Naive Bayes": "naive_bayes",
    "Support Vector Machine": "svm",
}

# Integer-coded columns whose values are category IDs, not quantities
# (see the UCI variable table). These are one-hot encoded for distance/linear models.
NOMINAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

BOM = chr(0xFEFF)


def clean_column_names(df):
    """The raw UCI file has stray whitespace (e.g. a tab) in some headers;
    Excel-saved CSVs may also carry a UTF-8 byte-order mark on the first one."""
    df.columns = [str(c).replace(BOM, "").strip() for c in df.columns]
    return df


def compute_metrics(y_true, y_pred, y_proba, labels):
    """Return the six assignment metrics for a multi-class problem.

    Precision, Recall, F1 and AUC (one-vs-rest) use weighted averaging so each
    class contributes in proportion to its support.
    """
    y_true = np.asarray(y_true)
    present = np.unique(y_true)
    if len(present) > 1:
        # roc_auc_score needs probability columns only for classes present in y_true
        col_idx = [list(labels).index(c) for c in present]
        proba = y_proba[:, col_idx]
        proba = proba / proba.sum(axis=1, keepdims=True)
        if len(present) == 2:
            auc = roc_auc_score(y_true == present[1], proba[:, 1])
        else:
            auc = roc_auc_score(
                y_true, proba, multi_class="ovr", average="weighted", labels=present
            )
    else:
        auc = float("nan")

    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
