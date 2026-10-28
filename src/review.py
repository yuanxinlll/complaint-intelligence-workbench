"""Validation-only threshold selection and consistent inference flags."""

import numpy as np

TARGET_ACCURACY = 0.80
MIN_ACCEPTED = 100
THRESHOLDS = [round(value / 1000, 3) for value in range(150, 976, 25)]


def select_review_rule(truth, predicted, scores):
    """Select maximum observed coverage meeting predefined validation criteria."""
    truth, predicted, scores = np.asarray(truth), np.asarray(predicted), np.asarray(scores)
    if not (len(truth) == len(predicted) == len(scores)) or not len(truth):
        raise ValueError("Validation arrays must be nonempty and equally sized.")
    curve = []
    for threshold in THRESHOLDS:
        accepted = scores >= threshold
        count = int(accepted.sum())
        accuracy = float((truth[accepted] == predicted[accepted]).mean()) if count else None
        curve.append({"threshold": threshold, "accepted_count": count, "coverage": count / len(truth), "accuracy": accuracy})
    eligible = [row for row in curve if row["accepted_count"] >= MIN_ACCEPTED and row["accuracy"] >= TARGET_ACCURACY]
    chosen = max(eligible, key=lambda row: (row["coverage"], -row["threshold"])) if eligible else None
    rule = {
        "mode": "threshold" if chosen else "review_all",
        "threshold": chosen["threshold"] if chosen else None,
        "selection_split": "validation", "target_observed_accuracy": TARGET_ACCURACY,
        "minimum_accepted_validation_rows": MIN_ACCEPTED,
        "selection_policy": "Maximum validation coverage satisfying predefined accuracy and sample count; ties use lower threshold; no eligible threshold means review all.",
        "validation_selection": chosen,
        "warning": "Exploratory validation-selected score rule, not a calibrated correctness probability or a production safety guarantee. Unknown categories may still get high scores.",
    }
    return rule, curve


def review_flags(scores, rule):
    values = np.asarray(scores, dtype=float)
    if rule["mode"] == "review_all":
        return np.ones(len(values), dtype=bool)
    if rule["mode"] != "threshold" or rule.get("threshold") is None:
        raise ValueError("Invalid review rule.")
    return ~np.isfinite(values) | (values < float(rule["threshold"]))
