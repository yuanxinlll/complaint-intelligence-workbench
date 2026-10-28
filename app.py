"""Stage 4 local complaint classification interface."""

from pathlib import Path
import hashlib

import pandas as pd
import streamlit as st

from src.interface import classify_batch, load_assets, read_upload
from src.predict import predict_text

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Complaint Intelligence Workbench", page_icon="📋", layout="wide")

if st.query_params.get("view") == "learn":
    st.markdown((ROOT / "LEARNING_GUIDE.md").read_text(encoding="utf-8"))
    st.link_button("打开分类工作台", "/")
    st.stop()

st.title("📋 Complaint Intelligence Workbench")
st.caption("Turn complaint narratives into issue categories for human review. A prototype for English debt collection complaints.")
st.link_button("How it works — Chinese learning guide", "/?view=learn")
st.warning("Reserved test accuracy: 48.50%. Accuracy among unflagged records: 70.77%. Unflagged predictions can still be wrong. Human judgment is required.")
st.caption("Text stays on this computer; no external model API is used. Scores are not correctness probabilities. Limited to 7 trained English complaint categories.")


@st.cache_resource
def assets(model_stamp, rule_stamp):
    return load_assets()


try:
    model_path, rule_path = ROOT / "artifacts/model.joblib", ROOT / "artifacts/review_rule.json"
    model, rule = assets(model_path.stat().st_mtime_ns, rule_path.stat().st_mtime_ns)
except Exception:
    st.error("The local model or review rule is missing, invalid, or mismatched. Follow the README to train and evaluate before reopening.")
    st.code("python -m src.train\npython -m src.evaluate")
    st.stop()

st.caption(f"Review rule: scores below {rule['threshold']} are flagged for human review." if rule["mode"] == "threshold" else "Review rule: flag every record for human review.")
single, batch = st.tabs(["Single complaint", "Batch CSV"])
with single:
    text = st.text_area("Complaint narrative (English)", height=160, max_chars=50000, placeholder="Describe what happened. No issue label is required.")
    st.caption("Maximum 50,000 characters. The model was trained on English complaints.")
    if st.button("Classify complaint", type="primary"):
        try:
            with st.spinner("Classifying locally…"):
                result = predict_text(text, model, rule)
            st.subheader("Prediction")
            st.write("Predicted issue:", result["predicted_issue"])
            st.write("Model score:", result["model_score"])
            st.write("Review status:", "Needs human review" if result["needs_review"] else "Not flagged by the rule; errors are still possible")
            st.write("Original narrative:", text)
        except ValueError as error:
            st.error(str(error))

with batch:
    st.write("Provide a narrative column, one complaint per row. Optional: complaint_id and issue. The issue column is shown as the original label and is never used for prediction. The official narrative header is also accepted.")
    st.caption("UTF-8 CSV only. Limits: 5 MB, 500 rows, 50,000 characters per narrative, 2 million characters total. Empty or oversized narratives remain in the output with an input error and no prediction.")
    example = pd.DataFrame({"complaint_id": ["demo-1", "demo-2", "demo-3"], "narrative": ["I do not owe this debt. The collector contacted the wrong person.", "I requested a written notice about the debt but have not received it.", "The collector keeps calling me at work several times every day."]})
    st.download_button("Download synthetic CSV example", example.to_csv(index=False).encode("utf-8-sig"), "synthetic_input_example.csv", "text/csv")
    upload = st.file_uploader("Choose a CSV file", type=["csv"], max_upload_size=5)
    demo = st.button("Try 3 synthetic examples")
    if demo:
        st.session_state.pop("upload_error", None)
        st.session_state["batch_input"] = example
        st.session_state["batch_origin"] = "3 synthetic examples for demonstration only; not evaluation data"
        st.session_state.pop("batch_result", None)
    if upload is not None:
        signature = (upload.name, hashlib.sha256(upload.getvalue()).hexdigest())
        if signature != st.session_state.get("upload_signature"):
            st.session_state["upload_signature"] = signature
            st.session_state.pop("batch_input", None)
            st.session_state.pop("batch_result", None)
            st.session_state.pop("upload_error", None)
            try:
                st.session_state["batch_input"] = read_upload(upload.getvalue())
                st.session_state["batch_origin"] = f"Uploaded file:{upload.name}"
            except ValueError as error:
                st.session_state["upload_error"] = str(error)
    if "upload_error" in st.session_state:
        st.error(st.session_state["upload_error"])
    if "batch_input" in st.session_state:
        frame = st.session_state["batch_input"]
        st.info(st.session_state["batch_origin"])
        duplicates = int(frame["narrative"].duplicated().sum())
        if duplicates:
            st.warning(f"Found {duplicates} duplicate narrative rows. Every row is retained and included in the batch statistics.")
        if st.button("Classify batch", type="primary"):
            with st.spinner(f"Classifying {len(frame)} records locally…"):
                st.session_state["batch_result"] = classify_batch(frame, model, rule)
    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        valid = result["input_error"].eq("")
        a, b, c = st.columns(3)
        a.metric("Batch records", len(result))
        b.metric("Classified", int(valid.sum()))
        c.metric("Input errors", int((~valid).sum()))
        st.caption("original_issue is the supplied label, if any; predicted_issue is the model output; needs_review is a rule-based hint.")
        category = st.selectbox("Filter by predicted issue", ["All"] + sorted(result.loc[valid, "predicted_issue"].unique().tolist()))
        review = st.selectbox("Filter by review status", ["All", "Needs human review", "Not flagged", "Input errors"])
        shown = result.copy()
        if category != "All":
            shown = shown.loc[shown["predicted_issue"].eq(category)]
        if review == "Needs human review":
            shown = shown.loc[shown["needs_review"] & shown["input_error"].eq("")]
        elif review == "Not flagged":
            shown = shown.loc[~shown["needs_review"]]
        elif review == "Input errors":
            shown = shown.loc[shown["input_error"].ne("")]
        st.dataframe(shown, hide_index=True, width="stretch")
        st.subheader("Predicted issue distribution — this batch, all valid rows")
        st.caption("The chart uses all valid rows in this batch and does not change with the table filters. Invalid rows are excluded. This is not the full CFPB database distribution.")
        if valid.any():
            counts = result.loc[valid, "predicted_issue"].value_counts().rename("count").to_frame()
            counts["proportion"] = counts["count"] / valid.sum()
            st.bar_chart(counts["count"])
            st.dataframe(counts, width="stretch")
        else:
            st.info("No valid predictions are available for the distribution chart.")
        st.download_button("Export filtered results as CSV", shown.to_csv(index=False).encode("utf-8-sig"), "complaint_predictions.csv", "text/csv")
