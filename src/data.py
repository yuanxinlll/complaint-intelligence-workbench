"""Minimal text cleaning and reproducible chronological splits."""

import hashlib
from pathlib import Path
import unicodedata

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN_END = "2024-12-12"
VALIDATION_END = "2024-12-21"
COLUMN_MAP = {"Complaint ID": "complaint_id", "Date received": "received_date", "Product": "product", "Issue": "issue", "Consumer complaint narrative": "narrative", "Company": "company"}


def clean_text(text):
    """Retain words, negation and digits; normalize Unicode and whitespace."""
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def prepare_data(raw):
    """Keep earliest observation of each normalized narrative, then split by date."""
    missing = set(COLUMN_MAP) - set(raw.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = raw[list(COLUMN_MAP)].rename(columns=COLUMN_MAP).copy()
    starting_rows = len(frame)
    valid = frame[["complaint_id", "received_date", "issue", "narrative"]].apply(lambda column: column.str.strip().ne("")).all(axis=1)
    frame = frame.loc[valid].copy()
    parsed = pd.to_datetime(frame["received_date"], format="%Y-%m-%d", errors="raise")
    if not parsed.between("2024-11-01", "2024-12-31").all():
        raise ValueError("Unexpected dates outside the agreed archive range.")
    if not frame["product"].eq("Debt collection").all():
        raise ValueError("Only Debt collection is supported in this model.")
    frame["_key"] = frame["narrative"].map(clean_text)
    frame = frame.loc[frame["_key"].ne("")].copy()
    invalid_rows = starting_rows - len(frame)
    frame = frame.sort_values(["received_date", "complaint_id"], kind="stable")
    duplicate_ids = int(frame["complaint_id"].duplicated().sum())
    frame = frame.drop_duplicates("complaint_id", keep="first")
    conflicting_groups = int((frame.groupby("_key")["issue"].nunique() > 1).sum())
    duplicate_texts = int(frame["_key"].duplicated().sum())
    frame = frame.drop_duplicates("_key", keep="first").copy()
    frame["text_hash"] = frame.pop("_key").map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest())
    splits = {
        "train": frame.loc[frame["received_date"].le(TRAIN_END)].reset_index(drop=True),
        "validation": frame.loc[frame["received_date"].gt(TRAIN_END) & frame["received_date"].le(VALIDATION_END)].reset_index(drop=True),
        "test": frame.loc[frame["received_date"].gt(VALIDATION_END)].reset_index(drop=True),
    }
    for name, part in splits.items():
        if part.empty:
            raise ValueError(f"Empty {name} split; review date boundaries.")
    for left, right in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if set(splits[left]["text_hash"]) & set(splits[right]["text_hash"]):
            raise AssertionError("Duplicate narrative leakage across splits.")
        if splits[left]["received_date"].max() >= splits[right]["received_date"].min():
            raise AssertionError("Chronological split overlap.")
    # Do not inspect test labels to choose classes or change the split.
    train_labels = sorted(splits["train"]["issue"].unique().tolist())
    unknown_validation = set(splits["validation"]["issue"]) - set(train_labels)
    if unknown_validation:
        raise ValueError(f"Validation has unseen labels: {sorted(unknown_validation)}")
    audit = {
        "input_rows": starting_rows, "invalid_or_empty_rows_removed": invalid_rows,
        "duplicate_id_rows_removed": duplicate_ids, "duplicate_narrative_rows_removed": duplicate_texts,
        "conflicting_narrative_groups_audited": conflicting_groups,
        "duplicate_rule": "Keep earliest received_date; tie-break by string complaint_id. Retain earliest label even when later labels conflict. Later labels never select the training label.",
        "normalization": "NFKC, whitespace collapse, casefold; no stop-word removal, no number or negation removal",
        "near_duplicate_limitation": "Only exact normalized duplicates removed; similar templates can remain.",
        "boundaries": {"train_end_inclusive": TRAIN_END, "validation_end_inclusive": VALIDATION_END, "test_end_inclusive": "2024-12-31"},
        "boundary_reason": "Fixed calendar windows: approximately first six weeks for training, next nine days for validation, last ten days reserved for independent testing; not optimized on metrics.",
        "split_rows": {name: len(part) for name, part in splits.items()},
        "split_date_ranges": {name: {"min": part["received_date"].min(), "max": part["received_date"].max()} for name, part in splits.items()},
        "train_class_counts": splits["train"]["issue"].value_counts().to_dict(),
        "validation_class_counts": splits["validation"]["issue"].value_counts().to_dict(),
        "class_exclusions": [], "labels_from_training": train_labels,
        "cross_split_duplicate_hashes": 0,
        "test_policy": "Test saved only; no test inference or metrics in stage 2.",
    }
    return splits, audit
