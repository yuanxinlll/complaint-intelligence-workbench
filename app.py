"""Local complaint classification assistant with explicit version selection."""
from pathlib import Path
import hashlib,json
import pandas as pd
import streamlit as st
from src.interface import classify_batch,read_upload
from src.predict import predict_text
from src.email_import import extract_email
from src.runtime_models import registry,load_profile
ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="Complaint Intelligence Workbench",page_icon="📋",layout="wide")
if st.query_params.get("view")=="learn":
    st.markdown((ROOT/"LEARNING_GUIDE_EN.md").read_text())
    st.link_button("Open workbench","/")
    st.stop()
if st.query_params.get("view")=="results":
    st.title("Recorded model results")
    st.dataframe(pd.DataFrame(json.loads((ROOT/"config/research_results.json").read_text())),hide_index=True,width="stretch")
    st.caption("Same repeatedly used 1,428-record development validation; different training versions. No fresh independent final test. Scores are not verified correctness probabilities.")
    st.link_button("Open workbench","/")
    st.stop()
st.title("📋 Complaint Intelligence Workbench")
st.caption("English debt-collection category suggestions for human review · Local processing")
profiles=registry();model_id=st.selectbox("Model",list(profiles["profiles"]),index=list(profiles["profiles"]).index(profiles["default"]),format_func=lambda key:profiles["profiles"][key]["label"])
@st.cache_resource
def assets(model_id,registry_stamp):return load_profile(model_id)
try:
    with st.spinner("Loading the local model…"):
        model,rule,info=assets(model_id,(ROOT/"config/model_profiles.json").stat().st_mtime_ns)
except Exception as error:
    st.error(str(error))
    st.info("A fresh clone contains no data or weights. See README for local artifact setup. No download, training or fallback was started automatically.")
    st.stop()
if st.session_state.get("active_model_version")!=info["version"]:
    st.session_state["active_model_version"]=info["version"]
    st.session_state.pop("batch_result",None)
    st.session_state.pop("eml_result",None)
st.caption(f"Active: {info['version']} · Development Accuracy {info['accuracy']:.2%} · Macro-F1 {info['macro_f1']:.4f}")
st.warning("Every suggestion needs human confirmation. Final confirmation decisions are not stored by this prototype.")
if model_id=="FT01":
    st.caption("Highest recorded development Macro-F1; lower Accuracy than DATA100 and higher runtime cost. Two small classes improve, while five class F1 scores decrease.")
    st.caption("Long complaints: the model uses the first and last 255 content tokens when over 510. Middle content is omitted; inspect the original text.")
st.link_button("How it works","/?view=learn")
st.link_button("Model results","/?view=results")
with st.expander("Model and review limitations"):
    st.write("All records require review; no acceptance threshold is transferred from an older model.")
    st.write("Possible ambiguity uses a fixed top-two score gap below 0.10. This unvalidated display heuristic is not a correctness probability; an absent hint does not guarantee correctness.")
    st.write("Validation has been repeatedly used. No new independent final test, measured enterprise time savings or verified email-domain accuracy is claimed. No external AI API or mailbox connection is used.")

def show_result(result, original):
    st.subheader("Suggested category")
    st.write(result["predicted_issue"])
    if result["possible_ambiguity"]:
        st.info("Possible ambiguity · Alternative: " + result["alternative_issue"])
    st.caption("Read the original text and confirm the category yourself. This suggestion is not a saved human decision.")
    if model_id == "FT01":
        details = model.input_details(original)
        if details["truncated"]:
            st.warning(f"Long text: {details['discarded_middle_tokens']} middle tokens were omitted. Check the original narrative before using this suggestion.")
    with st.expander("Scores and original text"):
        st.write("Model score:", round(result["model_score"], 4))
        st.caption("Uncalibrated score; not a verified correctness probability.")
        st.text(original)
        st.caption("Model version: " + info["version"])

single, email_tab, batch, overview = st.tabs(["Paste text", "Import email", "Batch CSV", "Project overview"])
with single:
    text = st.text_area("Complaint narrative (English)", height=160, max_chars=50000, placeholder="Describe what happened. No issue label is required.")
    st.caption("Maximum 50,000 characters. The model was trained on English complaints.")
    if st.button("Classify complaint", type="primary"):
        try:
            with st.spinner("Classifying locally…"):
                result = predict_text(text, model, rule)
            show_result(result, text)
        except ValueError as error:
            st.error(str(error))

with email_tab:
    st.write("Import an unlabeled English debt-collection email. No mailbox login or existing category is needed.")
    st.caption("Local .eml only · Maximum 5 MB · Attachments ignored · No remote images or links are opened.")
    st.download_button("Download a synthetic email example", b"Subject: Missing debt notice\r\nFrom: demo@example.com\r\nTo: reviewer@example.com\r\nMIME-Version: 1.0\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nI requested written proof of this debt but have not received a notice.\r\n", "synthetic_complaint.eml", "message/rfc822")
    mail = st.file_uploader("Choose an email file", type=["eml"], max_upload_size=5, key="eml_upload")
    if mail is not None:
        signature = hashlib.sha256(mail.getvalue()).hexdigest()
        if signature != st.session_state.get("eml_signature"):
            st.session_state["eml_signature"] = signature
            for key in ("eml_text", "eml_confirm", "eml_result", "eml_error", "eml_original"):
                st.session_state.pop(key, None)
            try:
                extracted = extract_email(mail.getvalue())
                st.session_state["eml_text"] = extracted["text"]
                st.session_state["eml_original"] = extracted
            except ValueError as error:
                st.session_state["eml_error"] = str(error)
        if st.session_state.get("eml_error"):
            st.error(st.session_state["eml_error"])
        elif "eml_original" in st.session_state:
            st.caption("Subject: " + (st.session_state["eml_original"]["subject"] or "(no subject)"))
            for warning in st.session_state["eml_original"]["warnings"]:
                st.caption(warning)
            with st.form("email_review_form"):
                reviewed = st.text_area("Review subject and body before classification", key="eml_text", height=240, max_chars=50000)
                checked = st.checkbox("I checked the extracted text and it is an English debt-collection complaint.", key="eml_confirm")
                submitted = st.form_submit_button("Classify reviewed email", type="primary")
            if submitted:
                if not checked:
                    st.warning("Check the extracted content before classifying.")
                else:
                    try:
                        result = predict_text(reviewed, model, rule)
                        st.session_state["eml_result"] = {"result":result,"text":reviewed}
                    except ValueError as error:
                        st.session_state.pop("eml_result", None)
                        st.error(str(error))
            if "eml_result" in st.session_state:
                saved = st.session_state["eml_result"]
                show_result(saved["result"], saved["text"])
                st.caption("Result applies to the last submitted text. Editing the email requires classifying it again.")
    st.caption("Email formatting differs from the complaint training data; email accuracy has not been evaluated. General mail is outside the seven-category task.")

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
                result_frame = st.session_state["batch_result"]
                result_frame["model_version"] = info["version"]
                result_frame["confirmation_status"] = "unconfirmed"
                result_frame["input_truncated"] = False
                if model_id == "FT01":
                    for row_index in result_frame.index[result_frame["input_error"].eq("")]:
                        result_frame.loc[row_index, "input_truncated"] = model.input_details(result_frame.loc[row_index, "narrative"])["truncated"]

    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        valid = result["input_error"].eq("")
        a, b, c = st.columns(3)
        a.metric("Batch records", len(result))
        b.metric("Classified", int(valid.sum()))
        c.metric("Input errors", int((~valid).sum()))
        if "possible_ambiguity" not in result:
            st.info("These cached results predate candidate display. Classify this batch again to view alternatives.")
        st.caption("predicted_issue is a suggestion, not a confirmed label. alternative_issue is shown only for the display ambiguity hint. Original labels remain separate; every result needs human confirmation.")
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
        compact_columns = [c for c in ["complaint_id", "predicted_issue", "alternative_issue", "input_error"] if c in shown]
        st.dataframe(shown[compact_columns], hide_index=True, width="stretch")
        with st.expander("Full batch details"):
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
        st.download_button("Export unconfirmed suggestions as CSV", shown.to_csv(index=False).encode("utf-8-sig"), "complaint_suggestions_unconfirmed.csv", "text/csv")

with overview:
    st.markdown((ROOT / "DEMO.md").read_text(encoding="utf-8"))
