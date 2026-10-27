"""Streamlit front-end: evaluate the pre-trained student-outcome classifiers on uploaded test data.

The app never retrains. It loads the fitted pipelines from model/ and scores the uploaded CSV.
"""

import io
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from model_utils import (
    METADATA_PATH,
    METRIC_NAMES,
    METRICS_PATH,
    MODEL_DIR,
    MODEL_FILES,
    TEST_DATA_PATH,
    clean_column_names,
    compute_metrics,
)

ACCENT = "#2a78d6"
INK_SECONDARY = "#52514e"
METRIC_HELP = {
    "Accuracy": "Share of students whose outcome was predicted correctly.",
    "AUC": "One-vs-rest ROC AUC, weighted by class support.",
    "Precision": "Weighted precision across Dropout / Enrolled / Graduate.",
    "Recall": "Weighted recall across Dropout / Enrolled / Graduate.",
    "F1": "Weighted harmonic mean of precision and recall.",
    "MCC": "Matthews correlation coefficient: -1 (inverse) to 1 (perfect), 0 = chance.",
}

# `python app.py` has no Streamlit runtime (only "missing ScriptRunContext" warnings);
# hand off to `streamlit run` so both launch styles open the app.
if __name__ == "__main__" and not st.runtime.exists():
    import sys

    from streamlit.web import cli as stcli

    sys.argv = ["streamlit", "run", __file__, *sys.argv[1:]]
    sys.exit(stcli.main())

st.set_page_config(page_title="Student Outcome Classifier Lab", page_icon="🎓", layout="wide")


# ---------- loading ----------

@st.cache_resource
def load_models():
    return {name: joblib.load(MODEL_DIR / f"{stem}.joblib") for name, stem in MODEL_FILES.items()}


@st.cache_data
def load_metadata():
    return json.loads(METADATA_PATH.read_text()), json.loads(METRICS_PATH.read_text())


@st.cache_data
def read_bundled_test_data():
    return TEST_DATA_PATH.read_bytes()


@st.cache_data
def parse_csv(raw_bytes):
    """Parse CSV bytes, sniffing the delimiter so ';'-separated files (like the raw UCI export) work."""
    return pd.read_csv(io.BytesIO(raw_bytes), sep=None, engine="python", encoding="utf-8-sig")


def prepare_uploaded_frame(raw_df, metadata):
    """Validate the uploaded CSV and return (X, y_encoded) or raise ValueError."""
    df = clean_column_names(raw_df.copy())
    features, target = metadata["feature_columns"], metadata["target_column"]
    labels = metadata["class_labels"]

    missing = [c for c in features + [target] if c not in df.columns]
    if missing:
        raise ValueError(
            f"{len(missing)} required column(s) missing: " + ", ".join(f"`{c}`" for c in missing)
        )

    X = df[features]
    non_numeric = [c for c in features if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError("Non-numeric values found in: " + ", ".join(f"`{c}`" for c in non_numeric))
    if X.isna().any().any():
        raise ValueError("Feature columns contain empty cells; please supply complete rows.")

    y_raw = df[target]
    if pd.api.types.is_numeric_dtype(y_raw) and set(y_raw.unique()) <= set(range(len(labels))):
        y = y_raw.astype(int)  # already encoded 0..n-1
    else:
        y_str = y_raw.astype(str).str.strip()
        unknown = sorted(set(y_str) - set(labels))
        if unknown:
            raise ValueError(f"Unknown target label(s) {unknown}; expected one of {labels}.")
        y = y_str.map({label: i for i, label in enumerate(labels)})
    return X, y.to_numpy()


@st.cache_data(show_spinner="Scoring all five models…")
def evaluate_all(X, y, _models, labels):
    """Score every model once per uploaded file; cached so switching models is instant."""
    out = {}
    class_ids = np.arange(len(labels))
    for name, model in _models.items():
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)
        out[name] = {
            "metrics": compute_metrics(y, y_pred, y_proba, class_ids),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    return out


# ---------- plotting ----------

def plot_confusion_matrix(cm, labels, normalise):
    data = cm / cm.sum(axis=1, keepdims=True) if normalise else cm
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(
        data, annot=True, fmt=".2f" if normalise else "d", cmap="Blues",
        xticklabels=labels, yticklabels=labels, cbar=False,
        linewidths=2, linecolor="white", annot_kws={"size": 11}, ax=ax,
    )
    ax.set_xlabel("Predicted outcome", color=INK_SECONDARY)
    ax.set_ylabel("Actual outcome", color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    fig.tight_layout()
    return fig


def plot_metric_bars(comparison, metric, highlight):
    series = comparison[metric].sort_values()
    colors = [ACCENT if name == highlight else "#b8cfee" for name in series.index]
    fig, ax = plt.subplots(figsize=(6.2, 2.9))
    ax.barh(series.index, series.values, color=colors, height=0.55)
    for i, v in enumerate(series.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9, color=INK_SECONDARY)
    lo = max(0.0, series.min() - 0.1) if metric != "MCC" else min(0.0, series.min() - 0.1)
    ax.set_xlim(lo, min(1.0, series.max() + 0.08))
    ax.set_xlabel(metric, color=INK_SECONDARY)
    ax.tick_params(colors=INK_SECONDARY, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d6d5d0")
    ax.grid(axis="x", color="#ecebe7", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ---------- page ----------

models = load_models()
metadata, holdout_metrics = load_metadata()
LABELS = metadata["class_labels"]

st.title("🎓 Student Outcome Classifier Lab")
st.caption(
    "Predicting whether a higher-education student will **drop out**, stay **enrolled**, or "
    "**graduate** - UCI dataset #697. Upload held-out data and compare five pre-trained models."
)

with st.sidebar:
    st.header("1 · Test data")
    uploaded = st.file_uploader(
        "Upload a test CSV",
        type=["csv"],
        help=f"Must contain the {len(metadata['feature_columns'])} feature columns and "
             f"`{metadata['target_column']}`. Models are not retrained on this file.",
    )
    st.download_button(
        "⬇ Download sample test_data.csv",
        data=read_bundled_test_data(),
        file_name="test_data.csv",
        mime="text/csv",
        width="stretch",
    )
    use_bundled = st.toggle(
        "Use bundled test_data.csv", value=uploaded is None, disabled=uploaded is not None,
        help="Handy for a quick look without uploading anything.",
    )

    st.header("2 · Model")
    selected_model = st.selectbox("Choose a classifier", list(MODEL_FILES))
    params = metadata["best_params"][selected_model]
    st.caption("Tuned hyper-parameters (5-fold CV):  \n" +
               "  \n".join(f"`{k}` = {v}" for k, v in params.items()))

if uploaded is not None:
    source_name = uploaded.name
    raw_bytes = uploaded.getvalue()
elif use_bundled:
    source_name = "test_data.csv (bundled)"
    raw_bytes = read_bundled_test_data()
else:
    st.info("⬅ Upload a test CSV or switch on **Use bundled test_data.csv** to begin.")
    st.stop()

try:
    raw_df = parse_csv(raw_bytes)
except Exception as exc:  # malformed / non-CSV upload
    st.error(f"**Could not read `{source_name}` as CSV.** {exc}")
    st.stop()

try:
    X, y = prepare_uploaded_frame(raw_df, metadata)
except ValueError as exc:
    st.error(f"**Could not evaluate `{source_name}`.** {exc}")
    st.stop()

results = evaluate_all(X, y, models, tuple(LABELS))
selected = results[selected_model]

class_counts = pd.Series(y).map(dict(enumerate(LABELS))).value_counts()
st.markdown(
    f"**Data:** `{source_name}` · {len(X):,} rows · "
    + " · ".join(f"{label}: {class_counts.get(label, 0):,}" for label in LABELS)
)

tab_model, tab_compare, tab_preds, tab_about = st.tabs(
    ["📊 Model evaluation", "🏁 Compare all models", "🔎 Predictions", "ℹ️ About"]
)

with tab_model:
    st.subheader(f"{selected_model}")
    cols = st.columns(len(METRIC_NAMES))
    for col, metric in zip(cols, METRIC_NAMES):
        value = selected["metrics"][metric]
        col.metric(metric, "n/a" if np.isnan(value) else f"{value:.4f}", help=METRIC_HELP[metric])

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown("**Confusion matrix**")
        normalise = st.toggle("Show row-normalised (recall per class)", value=False)
        cm = confusion_matrix(y, selected["y_pred"], labels=range(len(LABELS)))
        fig = plot_confusion_matrix(cm, LABELS, normalise)
        st.pyplot(fig, width="content")
        plt.close(fig)
    with right:
        st.markdown("**Classification report**")
        report = classification_report(
            y, selected["y_pred"], labels=range(len(LABELS)), target_names=LABELS,
            output_dict=True, zero_division=0,
        )
        report_df = pd.DataFrame(report).T
        report_df["support"] = report_df["support"].astype(int)
        st.dataframe(
            report_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}",
                                    "f1-score": "{:.3f}", "support": "{:,}"}),
            width="stretch",
        )
        worst = min(LABELS, key=lambda c: report[c]["recall"])
        st.caption(
            f"Hardest class for this model: **{worst}** "
            f"(recall {report[worst]['recall']:.2f})."
        )

with tab_compare:
    comparison = pd.DataFrame({n: r["metrics"] for n, r in results.items()}).T[METRIC_NAMES]
    st.subheader("All five models on this data")
    st.dataframe(
        comparison.style.format("{:.4f}").highlight_max(axis=0, color="#d4e4f7"),
        width="stretch",
    )
    st.caption("Highlighted cell = best model for that metric.")

    c1, c2 = st.columns([1, 2.4])
    with c1:
        chart_metric = st.radio("Metric to chart", METRIC_NAMES, index=METRIC_NAMES.index("F1"))
        leader = comparison[chart_metric].idxmax()
        st.metric(f"Best by {chart_metric}", leader, f"{comparison.loc[leader, chart_metric]:.4f}",
                  delta_color="off")
    with c2:
        fig = plot_metric_bars(comparison, chart_metric, selected_model)
        st.pyplot(fig, width="stretch")
        plt.close(fig)
        st.caption(f"Darker bar = currently selected model ({selected_model}).")

    with st.expander("Reference: metrics recorded at training time on test_data.csv"):
        st.dataframe(pd.DataFrame(holdout_metrics).T[METRIC_NAMES].style.format("{:.4f}"),
                     width="stretch")

with tab_preds:
    proba = selected["y_proba"]
    preds_df = pd.DataFrame({
        "Actual": [LABELS[i] for i in y],
        "Predicted": [LABELS[i] for i in selected["y_pred"]],
        "Confidence": proba.max(axis=1),
    })
    preds_df["Correct"] = preds_df["Actual"] == preds_df["Predicted"]
    for i, label in enumerate(LABELS):
        preds_df[f"P({label})"] = proba[:, i]

    show = st.radio("Show rows", ["All", "Misclassified only"], horizontal=True)
    view = preds_df if show == "All" else preds_df[~preds_df["Correct"]]
    st.caption(f"{(~preds_df['Correct']).sum():,} of {len(preds_df):,} rows misclassified by "
               f"{selected_model}.")
    st.dataframe(
        view, width="stretch", height=380,
        column_config={"Confidence": st.column_config.ProgressColumn(
            "Confidence", min_value=0.0, max_value=1.0, format="%.2f")},
    )
    st.download_button(
        "⬇ Download predictions", view.to_csv(index=False).encode(),
        file_name=f"predictions_{MODEL_FILES[selected_model]}.csv", mime="text/csv",
    )

with tab_about:
    st.markdown(
        f"""
**Dataset.** {metadata['dataset']} - 4,424 student records, 36 features
(demographic, socio-economic, macro-economic and 1st/2nd-semester academic performance)
and a 3-class target.

**Training protocol.** Stratified 80/20 split (`random_state={metadata['random_state']}`):
{metadata['train_rows']:,} rows for training with 5-fold stratified grid-search
(scoring = weighted F1), {metadata['test_rows']:,} rows held out as `test_data.csv`.

**Pre-processing** (inside each saved pipeline): nominal code columns such as *Course* or
*Mother's occupation* are one-hot encoded (codes seen < 10 times pooled), the remaining
columns are standardised. The Decision Tree uses raw features.

**Metric averaging.** Precision, Recall, F1 and one-vs-rest AUC are support-weighted
averages across the three classes; MCC uses its native multi-class form.
"""
    )
    st.markdown("**Expected input columns**")
    st.code(", ".join(metadata["feature_columns"] + [metadata["target_column"]]), language=None)
