"""Create a separate data candidate; never train or overwrite baseline files."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.data import clean_text

TRAIN_END = "2024-12-12"
VALIDATION_END = "2024-12-21"
VERSION = "candidate_v2_train_conflict_quarantine"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_hash(text):
    return hashlib.sha256(clean_text(text).encode("utf-8")).hexdigest()


def conflicts_known_by(raw, cutoff):
    """Use only labels received on/before cutoff, never later labels to relabel."""
    past = raw.loc[raw["Date received"].le(cutoff)].copy()
    past["text_hash"] = past["Consumer complaint narrative"].map(normalized_hash)
    group_counts = past.groupby("text_hash")["Issue"].nunique()
    conflict_hashes = set(group_counts[group_counts.gt(1)].index)
    return conflict_hashes, past.loc[past.text_hash.isin(conflict_hashes)].copy()


def main():
    if (OUT / "policy.json").exists() or (OUT / "data").exists():
        raise FileExistsError("Data version already started or exists. Preserve it; use a new version directory.")
    protected = [ROOT / name for name in ["app.py", "README.md", "PROJECT_STATUS.md", "STATUS.md", "LEARNING_GUIDE.md"]]
    for folder in ["src", "tests", "artifacts", "data", "reports", "diagnostics/stage_a_20261028", "diagnostics/stage_b_20261028"]:
        protected += [path for path in (ROOT / folder).rglob("*") if path.is_file() and not any(part in {"__pycache__", "matplotlib_cache", "plot_cache"} for part in path.parts)]
    before = {str(path.relative_to(ROOT)): sha(path) for path in protected}
    baseline = json.loads((ROOT / "artifacts/data_audit.json").read_text())
    baseline_meta = json.loads((ROOT / "artifacts/metadata.json").read_text())
    source = ROOT / "data/raw/debt_collection_2024_nov_dec.csv"
    if sha(source) != baseline_meta["raw_subset_sha256"]:
        raise ValueError("Raw source differs from the original run.")
    for name in ["train", "validation", "test"]:
        if sha(ROOT / f"data/processed/{name}.csv") != baseline["processed_sha256"][name]:
            raise ValueError(f"Baseline {name} differs from original run.")
    policy = {
        "version": VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Stage C data preparation only. No model training, evaluation, threshold selection, or app integration.",
        "baseline_model_sha256": sha(ROOT / "artifacts/model.joblib"),
        "input_source_sha256": sha(source),
        "normalization": "Existing src.data.clean_text: Unicode NFKC, whitespace collapse and casefold; retain negation, amounts, numbers and punctuation.",
        "exact_duplicate_policy": "Retain baseline global earliest received_date representative; same-day tie uses string complaint_id. Later duplicates remain excluded even if the earliest representative is quarantined.",
        "chronology": {"train_end_inclusive": TRAIN_END, "validation_end_inclusive": VALIDATION_END, "historical_holdout_end": "2024-12-31"},
        "conflict_training_policy": "Detect multiple Issue labels for an exact normalized text using ONLY raw records received on/before training end. Quarantine its earliest baseline training representative, preserving all original labels and sources. Do not resolve by majority or model output.",
        "later_label_policy": "A conflict first observed during validation does NOT retroactively remove a training record; it is recorded only in a separate ambiguity register.",
        "evaluation_population": "Copy baseline validation and historical holdout byte-for-byte; no ambiguity, error, template or score filtering of evaluation rows.",
        "near_template_policy": "Flag the existing stage B diagnostic candidate only. No automatic merge, deletion, relabeling or movement across time boundaries. Similarity threshold 0.90 is not an event-identity or data-cleaning rule.",
        "minority_policy": "Retain all seven classes; no class deletion, oversampling or invented text. Report every per-class removal.",
        "evaluation_status": "Validation already seen in A/B. Old test is a development-seen historical holdout, not untouched final evidence for new model selection.",
    }
    (OUT / "policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    raw = pd.read_csv(source, dtype=str, keep_default_na=False)
    raw = raw.loc[raw["Date received"].le(VALIDATION_END)].copy()
    if raw["Complaint ID"].duplicated().any():
        raise ValueError("Unexpected duplicate complaint IDs: investigate before this version can be built.")
    train = pd.read_csv(ROOT / "data/processed/train.csv", dtype=str, keep_default_na=False)
    validation = pd.read_csv(ROOT / "data/processed/validation.csv", dtype=str, keep_default_na=False)
    train_conflicts, training_conflict_records = conflicts_known_by(raw, TRAIN_END)
    known_validation, records_through_validation = conflicts_known_by(raw, VALIDATION_END)
    quarantined = train.loc[train.text_hash.isin(train_conflicts)].copy()
    revised = train.loc[~train.text_hash.isin(train_conflicts)].copy()
    assert len(quarantined) == len(train_conflicts)
    assert set(revised.issue) == set(train.issue)
    assert not revised.text_hash.duplicated().any()
    assert revised.received_date.le(TRAIN_END).all()
    assert train.text_hash.eq(train.narrative.map(normalized_hash)).all()
    assert validation.text_hash.eq(validation.narrative.map(normalized_hash)).all()
    # Re-verify earliest representative lineage using past records, not later labels.
    expected = raw.loc[raw["Date received"].le(TRAIN_END)].sort_values(["Date received", "Complaint ID"],kind="stable").copy()
    expected["text_hash"] = expected["Consumer complaint narrative"].map(normalized_hash)
    expected = expected.drop_duplicates("text_hash")
    assert expected["Complaint ID"].tolist() == train.complaint_id.tolist()
    raw_by_id = raw.set_index("Complaint ID")
    for original_name, processed_name in [("Consumer complaint narrative", "narrative"), ("Issue", "issue"), ("Date received", "received_date")]:
        assert revised[processed_name].tolist() == raw_by_id.loc[revised.complaint_id, original_name].tolist()
    data_dir = OUT / "data"
    data_dir.mkdir()
    revised.to_csv(data_dir / "train.csv", index=False, mode="x")
    quarantined.to_csv(data_dir / "quarantined_train.csv", index=False, mode="x")
    training_conflict_records.drop(columns=[]).to_csv(data_dir / "training_conflict_source_records.csv",index=False,mode="x")
    for name in ["validation", "test"]:
        shutil.copyfile(ROOT / f"data/processed/{name}.csv", data_dir / f"{name}.csv")
    # Future labels are not used. Test checks read only identity/date/text columns.
    holdout = pd.read_csv(data_dir / "test.csv", dtype=str, keep_default_na=False, usecols=["complaint_id", "received_date", "narrative", "text_hash"])
    assert holdout.text_hash.eq(holdout.narrative.map(normalized_hash)).all()
    cross = {}
    for left, right, a, b in [("train", "validation", revised, validation), ("train", "historical_test", revised, holdout), ("validation", "historical_test", validation, holdout)]:
        check = {"id_overlap": len(set(a.complaint_id)&set(b.complaint_id)), "exact_text_overlap": len(set(a.narrative)&set(b.narrative)), "normalized_overlap": len(set(a.text_hash)&set(b.text_hash)), "strict_chronology": bool(a.received_date.max() < b.received_date.min())}
        assert check["id_overlap"] == check["exact_text_overlap"] == check["normalized_overlap"] == 0 and check["strict_chronology"]
        cross[f"{left}__{right}"] = check
    registry = []
    for text_hash, group in records_through_validation.groupby("text_hash", sort=True):
        ordered = group.sort_values(["Date received", "Complaint ID"], kind="stable")
        first_label = ordered.iloc[0]["Issue"]
        discovered = ordered.loc[ordered.Issue.ne(first_label)].iloc[0]["Date received"]
        registry.append({"text_hash": text_hash, "first_conflict_observed_date": discovered, "training_cutoff_conflict": text_hash in train_conflicts, "action": "quarantined_training_representative" if text_hash in train_conflicts else "annotation_only_no_training_change", "source_complaint_ids": json.dumps(ordered["Complaint ID"].tolist()), "original_issues": json.dumps(sorted(group.Issue.unique().tolist())), "adjudicated_issue": "", "human_adjudicated": False})
    pd.DataFrame(registry).to_csv(OUT / "ambiguity_register.csv", index=False)
    records_through_validation.to_csv(OUT / "conflict_source_records_through_validation.csv", index=False)
    templates = json.loads((ROOT / "diagnostics/stage_b_20261028/template_checks.json").read_text())
    flagged = [row for row in templates if row["above_diagnostic_threshold_0_90"]]
    (OUT / "template_candidates.json").write_text(json.dumps({"source": "Existing stage B sample diagnostic; no new similarity search", "action": "flag_only_no_data_change", "candidates": flagged}, ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    counts = pd.DataFrame({"old_train": train.issue.value_counts(), "new_train": revised.issue.value_counts(), "quarantined_train": quarantined.issue.value_counts(), "validation_unchanged": validation.issue.value_counts()}).fillna(0).astype(int)
    counts.to_csv(OUT / "class_changes.csv", index_label="issue")
    copied_unchanged = {name: sha(data_dir/f"{name}.csv") == baseline["processed_sha256"][name] for name in ["validation","test"]}
    assert all(copied_unchanged.values())
    changed = [name for name,value in before.items() if sha(ROOT/name)!=value]
    assert not changed
    pd.testing.assert_frame_equal(pd.read_csv(data_dir/"train.csv",dtype=str,keep_default_na=False),revised.reset_index(drop=True))
    manifest = {
        "version": VERSION, "policy_sha256": sha(OUT/"policy.json"), "raw_subset_sha256": sha(source),
        "old_rows": {"train":len(train),"validation":len(validation),"historical_test":len(holdout)},
        "new_rows": {"train":len(revised),"validation":len(validation),"historical_test":len(holdout)},
        "quarantined_training_representatives":len(quarantined), "training_conflict_original_records":len(training_conflict_records),
        "conflict_groups_known_at_train_end":len(train_conflicts), "conflict_groups_known_at_validation_end":len(known_validation),
        "newly_observed_during_validation_not_used_for_training":len(known_validation-train_conflicts),
        "validation_ambiguity_rows_not_removed":int(validation.text_hash.isin(known_validation).sum()),
        "test_labels_used_for_policy":False, "test_inference_performed":False,
        "evaluation_files_byte_identical":copied_unchanged,"cross_split_checks":cross,
        "all_original_labels_preserved":True,"all_seven_classes_retained":True,
        "data_files":{str(path.relative_to(OUT)):{"bytes":path.stat().st_size,"sha256":sha(path)} for path in sorted(data_dir.glob("*.csv"))},
        "protected_files_sha256_before":before,"protected_files_unchanged":True,"changed_original_files":changed,
        "trained_models":0,"model_evaluations":0,"new_thresholds_selected":0,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({key:value for key,value in manifest.items() if key not in ["protected_files_sha256_before","data_files"]},indent=2,ensure_ascii=False))


if __name__ == "__main__":
    main()
