"""Streamlit demo for the production sentiment classifier.

Uses the SAME inference module as the FastAPI service and the
scripts/predict_svm.py CLI (src/inference_api.py) -- no duplicated
preprocessing or model-loading logic, and no retraining happens here.
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.inference_api import load_production_artifacts, predict_many

TEST_ACCURACY = 0.9072
MACRO_F1 = 0.9064
LABEL_STYLE = {
    "Positive": st.success,
    "Negative": st.error,
    "Neutral": st.info,
}
LABEL_EMOJI = {"Positive": "🟢", "Negative": "🔴", "Neutral": "🔵"}

st.set_page_config(page_title="Tweet Sentiment Classifier", page_icon="🐦")


@st.cache_resource
def get_artifacts():
    return load_production_artifacts()


def render_result(result: dict):
    label = result["predicted_label"]
    emoji = LABEL_EMOJI.get(label, "")

    st.caption("Predicted sentiment")
    LABEL_STYLE.get(label, st.write)(f"## {emoji} {label}")

    st.caption("Decision scores by class")
    scores_df = pd.DataFrame(
        {"class": list(result["decision_scores"].keys()),
         "decision_score": list(result["decision_scores"].values())}
    ).set_index("class")
    st.bar_chart(scores_df, height=320)

    col1, col2 = st.columns(2)
    col1.metric("Decision score (predicted class)", f"{result['decision_score_predicted_class']:.3f}")
    col2.metric("Confidence proxy (margin)", f"{result['confidence_proxy']:.3f}")


st.title("Tweet Sentiment Classifier")
st.write(
    "Classifies short text (e.g. tweets) as **Positive**, **Negative**, or **Neutral** using the "
    "project's selected production model."
)

st.markdown("**Model:** TF-IDF + Linear SVM")
info_col1, info_col2 = st.columns(2)
info_col1.metric("Test accuracy", f"{TEST_ACCURACY:.2%}")
info_col2.metric("Macro F1", f"{MACRO_F1:.2%}")

try:
    artifacts = get_artifacts()
    load_error = None
except Exception as exc:  # noqa: BLE001 -- surface any load failure in the UI, not a crash
    artifacts = None
    load_error = str(exc)

if load_error:
    st.error(
        f"Could not load the production model artifacts: {load_error}\n\n"
        "Make sure `artifacts/label_encoder.pkl`, `artifacts/baselines/tfidf_vectorizer.pkl`, and "
        "`artifacts/baselines/tfidf_linear_svm.pkl` exist (see README -- "
        "`python scripts/prepare_ci_artifacts.py` regenerates them if missing)."
    )
    st.stop()

st.divider()

mode = st.radio("Mode", ["Single text", "Batch (one per line)"], horizontal=True)

if mode == "Single text":
    text = st.text_area("Enter text to classify", placeholder="This product is absolutely amazing!")
    if st.button("Predict", type="primary"):
        if not text or not text.strip():
            st.warning("Please enter some text.")
        else:
            result = predict_many([text], artifacts)[0]
            render_result(result)

else:
    batch_text = st.text_area(
        "Enter one text per line",
        placeholder="This product is absolutely amazing!\nThe service was terrible.\nThe store opens at 9am.",
        height=150,
    )
    if st.button("Predict batch", type="primary"):
        lines = [line for line in batch_text.splitlines() if line.strip()]
        if not lines:
            st.warning("Please enter at least one non-empty line.")
        else:
            results = predict_many(lines, artifacts)
            table = pd.DataFrame([
                {
                    "text": r_text,
                    "predicted_label": r["predicted_label"],
                    "decision_score_predicted_class": round(r["decision_score_predicted_class"], 3),
                    "confidence_proxy": round(r["confidence_proxy"], 3),
                }
                for r_text, r in zip(lines, results)
            ])
            st.dataframe(table, use_container_width=True)
