"""CSV validation and local batch inference, shared with the Streamlit UI."""

import csv
import hashlib
import io
import json

import joblib
import pandas as pd

from src.data import ROOT, clean_text
from src.review import review_flags

MAX_BYTES = 5 * 1024 * 1024
MAX_ROWS = 500
MAX_TEXT_LENGTH = 50000
MAX_TOTAL_CHARACTERS = 2000000
OUTPUT_COLUMNS = {"predicted_issue", "model_score", "needs_review", "input_error"}


def load_assets():
    model_path = ROOT / "artifacts/model.joblib"
    rule_path = ROOT / "artifacts/review_rule.json"
    if not model_path.exists() or not rule_path.exists():
        raise ValueError("缺少本地模型或复核规则。请先按 README 完成训练和评估。")
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    if rule.get("model_sha256") != hashlib.sha256(model_path.read_bytes()).hexdigest():
        raise ValueError("模型和复核规则不匹配，请使用同一次训练、评估生成的文件。")
    review_flags([.5], rule)
    return joblib.load(model_path), rule


def read_upload(content):
    if len(content) > MAX_BYTES:
        raise ValueError("文件超过 5 MB，请先拆分。")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("文件编码无法读取，请在表格软件中另存为 CSV UTF-8。") from error
    try:
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        header = [name.strip() for name in header]
        if len(set(header)) != len(header) or any(not name for name in header):
            raise ValueError("列名不能为空或重复，请检查 CSV 第一行。")
        for number, row in enumerate(reader, start=2):
            if row and len(row) != len(header):
                raise ValueError(f"CSV 第 {number} 条记录的列数与表头不一致，请检查逗号和引号。")
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (StopIteration, pd.errors.EmptyDataError, pd.errors.ParserError, csv.Error) as error:
        raise ValueError("无法读取 CSV，请检查文件是否为空、分隔符和引号是否正确。") from error
    frame.columns = header
    if frame.empty or len(frame) > MAX_ROWS:
        raise ValueError("CSV 必须包含 1–500 条记录，请拆分后上传。")
    collision = OUTPUT_COLUMNS.intersection(frame.columns)
    if collision:
        raise ValueError(f"请移除上一次预测输出列再上传：{', '.join(sorted(collision))}")
    narrative_columns = [key for key in ["narrative", "Consumer complaint narrative"] if key in frame]
    if len(narrative_columns) != 1:
        raise ValueError("必须提供 narrative 列（或官方 Consumer complaint narrative 列），且二者只保留一个。")
    if narrative_columns[0] != "narrative":
        frame = frame.rename(columns={narrative_columns[0]: "narrative"})
    label_columns = [key for key in ["issue", "Issue", "original_issue"] if key in frame]
    if len(label_columns) > 1:
        raise ValueError("原始标签只能提供 issue、Issue、original_issue 其中一列。")
    if label_columns:
        frame = frame.rename(columns={label_columns[0]: "original_issue"})
    if frame["narrative"].str.len().sum() > MAX_TOTAL_CHARACTERS:
        raise ValueError("本批正文总长度超过 200 万字符，请拆成更小批次。")
    return frame


def classify_batch(frame, model, rule):
    result = frame.copy()
    errors = result["narrative"].map(lambda text: "正文为空" if not clean_text(text) else "正文超过 50,000 字符" if len(text) > MAX_TEXT_LENGTH else "")
    result["predicted_issue"] = ""
    result["model_score"] = float("nan")
    result["needs_review"] = True
    result["input_error"] = errors
    valid = errors.eq("")
    if valid.any():
        scores = model.predict_proba(result.loc[valid, "narrative"])
        result.loc[valid, "predicted_issue"] = model.classes_[scores.argmax(axis=1)]
        result.loc[valid, "model_score"] = scores.max(axis=1)
        result.loc[valid, "needs_review"] = review_flags(scores.max(axis=1), rule)
    return result
