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
st.caption("把公开投诉正文整理成问题类别，辅助人工浏览和复核。英文债务催收投诉学习原型。")
st.link_button("从零开始：我们到底做了什么？", "/?view=learn")
st.warning("独立测试 Accuracy 为 48.50%；达到复核门槛的部分也只有 70.77%。未标记复核不代表可靠，不能直接用于自动业务决定。")
st.caption("正文只在本机处理，不发送到外部模型。模型分数不是正确率。只适用于所训练的 7 个英文投诉类别。")


@st.cache_resource
def assets(model_stamp, rule_stamp):
    return load_assets()


try:
    model_path, rule_path = ROOT / "artifacts/model.joblib", ROOT / "artifacts/review_rule.json"
    model, rule = assets(model_path.stat().st_mtime_ns, rule_path.stat().st_mtime_ns)
except Exception:
    st.error("模型或复核规则不存在、损坏或不匹配。请按 README 完成训练和评估后重新打开。")
    st.code("python -m src.train\npython -m src.evaluate")
    st.stop()

st.caption(f"复核规则：分数低于 {rule['threshold']} 时建议人工复核。" if rule["mode"] == "threshold" else "复核规则：所有记录建议人工复核。")
single, batch = st.tabs(["输入一条投诉", "上传 CSV 批量分类"])
with single:
    text = st.text_area("英文投诉正文", height=160, max_chars=50000, placeholder="输入投诉的事情经过，不需要填写正确类别。")
    st.caption("最多 50,000 字符。中文界面不代表模型已学会中文投诉分类。")
    if st.button("开始单条分类", type="primary"):
        try:
            with st.spinner("正在本机分类…"):
                result = predict_text(text, model, rule)
            st.subheader("模型预测结果")
            st.write("预测类别：", result["predicted_issue"])
            st.write("模型分数：", result["model_score"])
            st.write("复核提示：", "建议人工复核" if result["needs_review"] else "规则未标记；仍可能分类错误")
            st.write("原始正文：", text)
        except ValueError as error:
            st.error(str(error))

with batch:
    st.write("CSV 必须有 narrative 列，每行一条投诉。可选 complaint_id 和 issue；issue 只作为原始标签展示，不进入预测。也支持官方正文字段名。")
    st.caption("UTF-8 CSV；最多 5 MB、500 行，每条 50,000 字符，总正文最多 200 万字符。空白或过长行保留并标明错误，不进行预测。")
    example = pd.DataFrame({"complaint_id": ["demo-1", "demo-2", "demo-3"], "narrative": ["I do not owe this debt. The collector contacted the wrong person.", "I requested a written notice about the debt but have not received it.", "The collector keeps calling me at work several times every day."]})
    st.download_button("下载格式示例（人为编写，非真实投诉）", example.to_csv(index=False).encode("utf-8-sig"), "synthetic_input_example.csv", "text/csv")
    upload = st.file_uploader("选择 CSV 文件", type=["csv"], max_upload_size=5)
    demo = st.button("用 3 条演示文本体验（非真实评估）")
    if demo:
        st.session_state.pop("upload_error", None)
        st.session_state["batch_input"] = example
        st.session_state["batch_origin"] = "人为编写的 3 条演示文本，不是真实评估数据"
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
                st.session_state["batch_origin"] = f"上传文件：{upload.name}"
            except ValueError as error:
                st.session_state["upload_error"] = str(error)
    if "upload_error" in st.session_state:
        st.error(st.session_state["upload_error"])
    if "batch_input" in st.session_state:
        frame = st.session_state["batch_input"]
        st.info(st.session_state["batch_origin"])
        duplicates = int(frame["narrative"].duplicated().sum())
        if duplicates:
            st.warning(f"发现 {duplicates} 条完全相同的正文超额记录。本次保留每一行，统计也会包含它们。")
        if st.button("开始批量分类", type="primary"):
            with st.spinner(f"正在本机处理 {len(frame)} 条…"):
                st.session_state["batch_result"] = classify_batch(frame, model, rule)
    if "batch_result" in st.session_state:
        result = st.session_state["batch_result"]
        valid = result["input_error"].eq("")
        a, b, c = st.columns(3)
        a.metric("本批总记录", len(result))
        b.metric("成功分类", int(valid.sum()))
        c.metric("输入异常", int((~valid).sum()))
        st.caption("original_issue（如有）是原始标签；predicted_issue 是模型预测；needs_review 是规则提示。")
        category = st.selectbox("按预测类别筛选", ["全部"] + sorted(result.loc[valid, "predicted_issue"].unique().tolist()))
        review = st.selectbox("按复核状态筛选", ["全部", "建议人工复核", "规则未标记", "输入异常"])
        shown = result.copy()
        if category != "全部":
            shown = shown.loc[shown["predicted_issue"].eq(category)]
        if review == "建议人工复核":
            shown = shown.loc[shown["needs_review"] & shown["input_error"].eq("")]
        elif review == "规则未标记":
            shown = shown.loc[~shown["needs_review"]]
        elif review == "输入异常":
            shown = shown.loc[shown["input_error"].ne("")]
        st.dataframe(shown, hide_index=True, width="stretch")
        st.subheader("本上传／演示批次的预测类别分布（全部有效行）")
        st.caption("图表不随表格筛选改变；异常行不计入类别比例。不是整个 CFPB 数据库的分布。")
        if valid.any():
            counts = result.loc[valid, "predicted_issue"].value_counts().rename("count").to_frame()
            counts["proportion"] = counts["count"] / valid.sum()
            st.bar_chart(counts["count"])
            st.dataframe(counts, width="stretch")
        else:
            st.info("没有有效预测，无法显示类别分布。")
        st.download_button("导出当前筛选结果 CSV", shown.to_csv(index=False).encode("utf-8-sig"), "complaint_predictions.csv", "text/csv")
