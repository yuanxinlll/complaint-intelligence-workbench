"""Stage 3: freeze a validation review rule, then evaluate the reserved test set."""

from datetime import datetime, timezone
import hashlib
import json
import os

import joblib
import numpy as np
import pandas as pd

from src.data import ROOT
from src.review import review_flags, select_review_rule
from src.train import validation_metrics

ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def save_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def progress(message):
    line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with (ARTIFACTS / "evaluation.log").open("a", encoding="utf-8") as output:
        output.write(line + "\n")


def review_performance(truth, predicted, scores, rule, labels):
    truth, predicted = np.asarray(truth), np.asarray(predicted)
    flagged = review_flags(scores, rule)
    accepted = ~flagged
    mistakes = predicted != truth
    return {
        "rows": len(truth), "accepted_count": int(accepted.sum()), "needs_review_count": int(flagged.sum()),
        "automatic_coverage": float(accepted.mean()),
        "accepted_metrics": validation_metrics(truth[accepted], predicted[accepted], labels) if accepted.any() else None,
        "reviewed_subset_accuracy": float((~mistakes[flagged]).mean()) if flagged.any() else None,
        "fraction_of_errors_flagged": float(flagged[mistakes].mean()) if mistakes.any() else None,
        "unflagged_errors": int((mistakes & accepted).sum()),
        "accepted_macro_f1_note": "Macro-F1 uses all 7 labels, including zero-support classes; check per-class support because accepted coverage can be concentrated in a few classes.",
    }


def draw_figures(metrics, curve, rule, review):
    os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / "matplotlib_cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["Debt not owed", "Communication tactics", "Electronic communication", "False statements", "Contact/share threats", "Negative/legal action", "Written notification"]
    matrix = np.asarray(metrics["confusion_matrix_true_rows_predicted_columns"])
    fig, axis = plt.subplots(figsize=(10, 8))
    display = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(7), labels, rotation=40, ha="right")
    axis.set_yticks(range(7), labels)
    axis.set(xlabel="Predicted label", ylabel="Consumer-selected label", title="Reserved test set: confusion matrix (counts)")
    for row in range(7):
        for col in range(7):
            axis.text(col, row, str(matrix[row, col]), ha="center", va="center", color="white" if matrix[row, col] > matrix.max() / 2 else "black")
    fig.colorbar(display, ax=axis)
    fig.tight_layout()
    fig.savefig(REPORTS / "test_confusion_matrix.png", dpi=150)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot([row["threshold"] for row in curve], [row["coverage"] for row in curve], "o-", label="Validation coverage")
    axis.plot([row["threshold"] for row in curve], [row["accuracy"] if row["accuracy"] is not None else np.nan for row in curve], "s-", label="Validation accepted accuracy")
    axis.axhline(.8, linestyle=":", color="gray", label="Observed accuracy target")
    if rule["threshold"] is not None:
        axis.axvline(rule["threshold"], linestyle="--", color="black", label=f"Frozen threshold {rule['threshold']}")
    axis.set(xlabel="Maximum model class score threshold", ylabel="Fraction", ylim=(0, 1.05), title="Validation trade-off: coverage vs observed accuracy")
    axis.legend(loc="best")
    fig.tight_layout()
    fig.savefig(REPORTS / "validation_review_curve.png", dpi=150)
    plt.close(fig)


def main():
    if (ARTIFACTS / "test_metrics.json").exists() or (ARTIFACTS / "review_rule.json").exists():
        raise FileExistsError("Evaluation already started or completed. Preserve its frozen rule and results.")
    progress("[1/5] 检查文件校验值，加载冻结模型；不重新训练。")
    audit = json.loads((ARTIFACTS / "data_audit.json").read_text())
    model_path = ARTIFACTS / "model.joblib"
    model = joblib.load(model_path)
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    labels = model.classes_.tolist()
    for split in ["validation", "test"]:
        path = ROOT / f"data/processed/{split}.csv"
        if hashlib.sha256(path.read_bytes()).hexdigest() != audit["processed_sha256"][split]:
            raise ValueError(f"{split} file changed since training.")
    validation = pd.read_csv(ROOT / "data/processed/validation.csv", dtype=str, keep_default_na=False)
    progress("[2/5] 仅使用验证集选门槛：准确率至少 80%，接受样本至少 100，优先覆盖最多。")
    validation_predicted = model.predict(validation["narrative"])
    validation_scores = model.predict_proba(validation["narrative"]).max(axis=1)
    rule, curve = select_review_rule(validation["issue"], validation_predicted, validation_scores)
    rule.update({"created_at_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": model_sha})
    save_json(ARTIFACTS / "review_rule.json", rule)
    save_json(ARTIFACTS / "validation_review_curve.json", curve)
    validation_review = review_performance(validation["issue"], validation_predicted, validation_scores, rule, labels)
    save_json(ARTIFACTS / "validation_review_metrics.json", validation_review)
    progress(f"规则已落盘冻结：mode={rule['mode']}，threshold={rule['threshold']}，验证覆盖率={validation_review['automatic_coverage']:.2%}。")
    progress("[3/5] 首次评估保留测试集；本次结果不会用于修改模型或门槛。")
    test = pd.read_csv(ROOT / "data/processed/test.csv", dtype=str, keep_default_na=False)
    if set(test["issue"]) - set(labels):
        raise ValueError("Test has unseen labels; report this before evaluation.")
    predicted = model.predict(test["narrative"])
    scores = model.predict_proba(test["narrative"]).max(axis=1)
    metrics = validation_metrics(test["issue"], predicted, labels)
    dummy = joblib.load(ARTIFACTS / "baseline.joblib")
    baseline = validation_metrics(test["issue"], dummy.predict(test["narrative"]), labels)
    review = review_performance(test["issue"], predicted, scores, rule, labels)
    save_json(ARTIFACTS / "test_metrics.json", {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": model_sha, "evaluation_split": "reserved_test", "baseline": baseline, "text_model": metrics, "review": review, "review_rule": rule, "test_used_for_selection": False})
    progress(f"测试 Accuracy={metrics['accuracy']:.4f}，Macro-F1={metrics['macro_f1']:.4f}，规则覆盖率={review['automatic_coverage']:.2%}。")
    progress("[4/5] 保存各类指标、原文关联、错误案例与混淆矩阵。")
    results = test[["complaint_id", "received_date", "issue", "narrative"]].copy()
    results["predicted_issue"] = predicted
    results["model_score"] = scores
    results["needs_review"] = review_flags(scores, rule)
    results.to_csv(ARTIFACTS / "test_predictions.csv", index=False)
    errors = results.loc[results["issue"].ne(results["predicted_issue"])].sort_values(["model_score", "complaint_id"], ascending=[False, True])
    # Deterministic examples expose high-score errors rather than selecting successes.
    examples = errors.head(5).copy()
    examples["narrative_excerpt"] = examples.pop("narrative").map(lambda text: " ".join(text.split())[:350] + "…")
    save_json(ARTIFACTS / "error_examples.json", {"selection": "Top 5 highest-score errors on reserved test; not representative of all errors.", "examples": json.loads(examples.to_json(orient="records"))})
    REPORTS.mkdir(exist_ok=True)
    draw_figures(metrics, curve, rule, review)
    progress("[5/5] 更新模型元信息；阶段 3 完成，阶段 4 未开始。")
    metadata_path = ARTIFACTS / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update({"review_threshold": rule["threshold"], "review_rule_mode": rule["mode"], "review_rule_status": "Frozen from validation; see review_rule.json", "review_rule_file": "review_rule.json", "test_evaluated": True, "test_metrics": "test_metrics.json"})
    save_json(metadata_path, metadata)


if __name__ == "__main__":
    main()
