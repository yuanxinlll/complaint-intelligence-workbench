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
        raise ValueError("The local model or review rule is missing. Complete training and evaluation following the README.")
    rule = json.loads(rule_path.read_text(encoding="utf-8"))
    if rule.get("model_sha256") != hashlib.sha256(model_path.read_bytes()).hexdigest():
        raise ValueError("The model and review rule do not match. Use files from the same training and evaluation run.")
    review_flags([.5], rule)
    return joblib.load(model_path), rule


def read_upload(content):
    if len(content) > MAX_BYTES:
        raise ValueError("The file exceeds 5 MB. Split it into smaller files.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError("Unsupported encoding. Save the file as CSV UTF-8 and try again.") from error
    try:
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        header = [name.strip() for name in header]
        if len(set(header)) != len(header) or any(not name for name in header):
            raise ValueError("Column names must be nonempty and unique. Check the CSV header.")
        for number, row in enumerate(reader, start=2):
            if row and len(row) != len(header):
                raise ValueError(f"CSV record {number} has a different number of columns from the header. Check commas and quotation marks.")
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)
    except (StopIteration, pd.errors.EmptyDataError, pd.errors.ParserError, csv.Error) as error:
        raise ValueError("Cannot read the CSV. Check for empty content, invalid delimiters, or unmatched quotes.") from error
    frame.columns = header
    if frame.empty or len(frame) > MAX_ROWS:
        raise ValueError("The CSV must contain 1–500 records. Split larger files before uploading.")
    collision = OUTPUT_COLUMNS.intersection(frame.columns)
    if collision:
        raise ValueError(f"Remove previous prediction output columns before uploading: {', '.join(sorted(collision))}")
    narrative_columns = [key for key in ["narrative", "Consumer complaint narrative"] if key in frame]
    if len(narrative_columns) != 1:
        raise ValueError("Provide exactly one narrative column: narrative or Consumer complaint narrative.")
    if narrative_columns[0] != "narrative":
        frame = frame.rename(columns={narrative_columns[0]: "narrative"})
    label_columns = [key for key in ["issue", "Issue", "original_issue"] if key in frame]
    if len(label_columns) > 1:
        raise ValueError("Provide at most one original label column: issue, Issue, or original_issue.")
    if label_columns:
        frame = frame.rename(columns={label_columns[0]: "original_issue"})
    if frame["narrative"].str.len().sum() > MAX_TOTAL_CHARACTERS:
        raise ValueError("Total narrative length exceeds 2 million characters. Split this batch.")
    return frame


def classify_batch(frame, model, rule):
    result = frame.copy()
    errors = result["narrative"].map(lambda text: "Empty narrative" if not clean_text(text) else "Narrative exceeds 50,000 characters" if len(text) > MAX_TEXT_LENGTH else "")
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
