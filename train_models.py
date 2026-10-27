"""Train and persist the five classifiers on the UCI Student Dropout dataset.

Run:  python train_models.py

Outputs
-------
test_data.csv                 stratified 20% hold-out (features + Target) for the Streamlit app
model/<name>.joblib           fitted sklearn Pipelines (preprocessing + estimator)
model/model_metadata.json     feature order, class labels, chosen hyper-parameters
model/metrics_holdout.json    evaluation metrics of every model on test_data.csv
"""

import json
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from model_utils import (
    METADATA_PATH,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    NOMINAL_COLUMNS,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_DATA_PATH,
    clean_column_names,
    compute_metrics,
)


def build_preprocessor(feature_columns):
    """One-hot encode category-ID columns, standardise everything else."""
    numeric_columns = [c for c in feature_columns if c not in NOMINAL_COLUMNS]
    return ColumnTransformer(
        transformers=[
            (
                "nominal",
                # Rare codes (<10 students) are pooled into one "infrequent" column
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=10,
                    sparse_output=False,
                ),
                NOMINAL_COLUMNS,
            ),
            ("numeric", StandardScaler(), numeric_columns),
        ]
    )


def build_search_spaces(feature_columns):
    """Pipeline + small hyper-parameter grid per model, tuned with 5-fold CV."""
    prep = build_preprocessor
    return {
        "Logistic Regression": (
            Pipeline([
                ("prep", prep(feature_columns)),
                ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [0.01, 0.1, 1.0, 10.0], "clf__class_weight": [None, "balanced"]},
        ),
        "Decision Tree": (
            # Trees are scale-invariant and split integer codes directly, so no preprocessing
            Pipeline([("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))]),
            {
                "clf__criterion": ["gini", "entropy"],
                "clf__max_depth": [4, 6, 8, 12, None],
                "clf__min_samples_leaf": [1, 5, 20],
            },
        ),
        "KNN": (
            Pipeline([("prep", prep(feature_columns)), ("clf", KNeighborsClassifier())]),
            {
                "clf__n_neighbors": [5, 11, 21, 31],
                "clf__weights": ["uniform", "distance"],
                "clf__p": [1, 2],
            },
        ),
        "Naive Bayes": (
            Pipeline([("prep", prep(feature_columns)), ("clf", GaussianNB())]),
            {"clf__var_smoothing": [1e-9, 1e-6, 1e-3, 1e-2, 1e-1]},
        ),
        "Support Vector Machine": (
            Pipeline([
                ("prep", prep(feature_columns)),
                ("clf", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]),
            {"clf__C": [0.5, 1.0, 5.0], "clf__gamma": ["scale", 0.01]},
        ),
    }


def main():
    df = clean_column_names(pd.read_csv(RAW_DATA_PATH, sep=";"))
    feature_columns = [c for c in df.columns if c != TARGET_COLUMN]
    print(f"Dataset: {df.shape[0]} rows, {len(feature_columns)} features")
    print(df[TARGET_COLUMN].value_counts().to_string(), "\n")

    # Models are trained on integer codes: sklearn 1.7's fast KNN path (uniform
    # weights + manhattan) casts class labels to int and fails on strings.
    class_labels = sorted(df[TARGET_COLUMN].unique().tolist())
    X = df[feature_columns]
    y = df[TARGET_COLUMN].map({label: i for i, label in enumerate(class_labels)})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    # The shared test file keeps human-readable labels
    test_df = X_test.copy()
    test_df[TARGET_COLUMN] = [class_labels[i] for i in y_test]
    test_df.to_csv(TEST_DATA_PATH, index=False)
    print(f"Saved hold-out set ({len(test_df)} rows) -> {TEST_DATA_PATH.name}")

    MODEL_DIR.mkdir(exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, best_params = {}, {}

    for name, (pipeline, grid) in build_search_spaces(feature_columns).items():
        start = time.perf_counter()
        search = GridSearchCV(pipeline, grid, scoring="f1_weighted", cv=cv, n_jobs=-1)
        search.fit(X_train, y_train)
        model = search.best_estimator_

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        results[name] = compute_metrics(y_test, y_pred, y_proba, model.classes_)
        best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}

        joblib.dump(model, MODEL_DIR / f"{MODEL_FILES[name]}.joblib")
        print(
            f"{name:<24} CV F1={search.best_score_:.4f}  "
            f"test F1={results[name]['F1']:.4f}  ({time.perf_counter() - start:.1f}s)  "
            f"{best_params[name]}"
        )

    metadata = {
        "dataset": "UCI #697 - Predict Students' Dropout and Academic Success",
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "class_labels": class_labels,  # list index == encoded class used by the models
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "random_state": RANDOM_STATE,
        "best_params": best_params,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2, default=str))
    METRICS_PATH.write_text(json.dumps(results, indent=2))

    table = pd.DataFrame(results).T.round(4)
    print("\nHold-out evaluation\n" + table.to_string())


if __name__ == "__main__":
    main()
