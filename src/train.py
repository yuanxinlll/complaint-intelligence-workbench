"""Stage 2: train once, report validation results, reserve test evaluation."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
import time
import warnings

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

from src.data import ROOT, clean_text, prepare_data

SEED = 42
RAW = ROOT / "data/raw/debt_collection_2024_nov_dec.csv"
ARTIFACTS = ROOT / "artifacts"
LOG = ARTIFACTS / "training.log"


def log(message):
    line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as output:
        output.write(line + "\n")


def validation_metrics(truth, predicted, labels):
    return {
        "accuracy": accuracy_score(truth, predicted),
        "macro_f1": f1_score(truth, predicted, labels=labels, average="macro", zero_division=0),
        "per_class": classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix_labels": labels,
        "confusion_matrix_true_rows_predicted_columns": confusion_matrix(truth, predicted, labels=labels).tolist(),
    }


def save_json(name, value):
    (ARTIFACTS / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if (ARTIFACTS / "model.joblib").exists() or LOG.exists():
        raise FileExistsError("Training artifacts already exist. Preserve this run; use a project copy for reruns.")
    started = time.monotonic()
    log("[1/6] 读取本地 CSV；正文不会写入训练日志。")
    raw = pd.read_csv(RAW, dtype=str, keep_default_na=False)
    log(f"读取完成：{len(raw):,} 条原始候选记录。")
    log("[2/6] 必要清洗、保留最早重复正文、按固定日期划分。")
    splits, audit = prepare_data(raw)
    processed = ROOT / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    for name, part in splits.items():
        output = processed / f"{name}.csv"
        if output.exists():
            raise FileExistsError(output)
        part.to_csv(output, index=False, mode="x")
        audit.setdefault("processed_sha256", {})[name] = hashlib.sha256(output.read_bytes()).hexdigest()
        log(f"{name}: {len(part):,} 条，{part['received_date'].min()} 至 {part['received_date'].max()}")
    save_json("data_audit.json", audit)
    train, validation = splits["train"], splits["validation"]
    labels = audit["labels_from_training"]
    log(f"删除重复正文 {audit['duplicate_narrative_rows_removed']:,} 条；跨集合重复正文 0；测试集不参与训练或评估。")
    log("[3/6] 训练 DummyClassifier：始终预测训练集最多的类别。")
    dummy = DummyClassifier(strategy="most_frequent", random_state=SEED)
    dummy.fit(train["narrative"], train["issue"])
    baseline = validation_metrics(validation["issue"], dummy.predict(validation["narrative"]), labels)
    log(f"基线验证 Accuracy={baseline['accuracy']:.4f}，Macro-F1={baseline['macro_f1']:.4f}")
    log("[4/6] 训练 TF-IDF + LogisticRegression；TF-IDF 仅在训练集拟合。")
    model = Pipeline([
        ("tfidf", TfidfVectorizer(preprocessor=clean_text, lowercase=False, token_pattern=r"(?u)\b\w+\b", ngram_range=(1, 2), min_df=2, max_df=0.98, max_features=40000, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000, solver="lbfgs", random_state=SEED)),
    ])
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(train["narrative"], train["issue"])
    log(f"训练完成：词表 {len(model.named_steps['tfidf'].vocabulary_):,} 项，迭代 {model.named_steps['classifier'].n_iter_.tolist()}。")
    log("[5/6] 仅预测验证集；不选择人工复核阈值。")
    predicted = model.predict(validation["narrative"])
    scores = model.predict_proba(validation["narrative"]).max(axis=1)
    metrics = validation_metrics(validation["issue"], predicted, labels)
    log(f"文本模型验证 Accuracy={metrics['accuracy']:.4f}，Macro-F1={metrics['macro_f1']:.4f}")
    predictions = validation[["complaint_id", "received_date", "issue"]].copy()
    predictions["predicted_issue"] = predicted
    predictions["model_score"] = scores
    predictions.to_csv(ARTIFACTS / "validation_predictions.csv", index=False)
    save_json("validation_metrics.json", {"evaluation_split": "validation", "baseline": baseline, "text_model": metrics, "review_threshold": None, "test_evaluated": False})
    log("[6/6] 保存模型、参数和依赖版本，并验证重新加载后的预测一致。")
    joblib.dump(model, ARTIFACTS / "model.joblib")
    joblib.dump(dummy, ARTIFACTS / "baseline.joblib")
    restored = joblib.load(ARTIFACTS / "model.joblib")
    if not (restored.predict(validation["narrative"].head(10)) == predicted[:10]).all():
        raise AssertionError("Saved model predictions do not match.")
    archive = json.loads((ROOT / "reports/archive_report.json").read_text())
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_source": archive["source"], "data_acquisition_date": "2026-10-28 (user supplied; exact download time unknown)",
        "raw_subset_sha256": hashlib.sha256(RAW.read_bytes()).hexdigest(),
        "filter": {"product": "Debt collection", "date_min": "2024-11-01", "date_max": "2024-12-31", "narrative_and_issue_required": True},
        "input_features": ["narrative"], "target": "issue", "labels": labels,
        "random_seed": SEED, "data_audit": "data_audit.json",
        "sample_counts": audit["split_rows"], "split_date_ranges": audit["split_date_ranges"],
        "model_parameters": {"tfidf": {"ngram_range": [1, 2], "min_df": 2, "max_df": .98, "max_features": 40000, "sublinear_tf": True, "token_pattern": r"(?u)\b\w+\b", "preprocessor": "src.data.clean_text"}, "classifier": {"C": 1.0, "class_weight": "balanced", "max_iter": 1000, "solver": "lbfgs", "random_state": SEED}},
        "versions": {name: importlib.metadata.version(name) for name in ["pandas", "numpy", "scikit-learn", "scipy", "joblib", "streamlit"]},
        "python": platform.python_version(), "validation_metrics": "validation_metrics.json",
        "review_threshold": None, "review_rule_status": "Not selected; stage 3 requires validation evidence.",
        "test_evaluated": False, "score_warning": "Maximum class score is not a calibrated probability of correctness and does not reliably detect unknown categories.",
        "checks": {"chronological_order": True, "cross_split_duplicates": 0, "saved_model_reload_predictions_match": True},
        "duration_seconds": round(time.monotonic() - started, 2),
    }
    save_json("metadata.json", metadata)
    versions = metadata["versions"]
    (ROOT / "requirements.txt").write_text("\n".join(f"{name}=={versions[name]}" for name in ["pandas", "numpy", "scikit-learn", "scipy", "joblib", "streamlit"]) + "\n", encoding="utf-8")
    log(f"阶段 2 完成，耗时 {metadata['duration_seconds']} 秒。停止；阶段 3 尚未开始。")


if __name__ == "__main__":
    main()
