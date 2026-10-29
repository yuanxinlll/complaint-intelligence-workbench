"""Predict a single complaint locally, without uploading its text."""

import argparse
import hashlib
import json

import joblib
import numpy as np

from src.data import ROOT, clean_text
from src.review import review_flags

# Display-only heuristic, not a fitted threshold or a correctness guarantee.
AMBIGUITY_GAP = 0.10


def validate_probabilities(scores, classes, expected_rows):
    """Reject malformed probability output before selecting any category."""
    values = np.asarray(scores, dtype=float)
    labels = np.asarray(classes)
    if labels.ndim != 1 or len(labels) < 2 or len(set(labels.tolist())) != len(labels):
        raise ValueError("The model must provide distinct category labels.")
    if values.shape != (expected_rows, len(labels)):
        raise ValueError("Model scores do not match the input rows and categories.")
    if not np.isfinite(values).all():
        raise ValueError("The model returned non-finite scores; no suggestion was produced.")
    if (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-5, rtol=1e-5):
        raise ValueError("The model returned invalid probability scores; no suggestion was produced.")
    return values


def candidate_details(scores, classes):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1:
        raise ValueError("Expected one probability score per category.")
    values = validate_probabilities(values[None, :], classes, 1)[0]
    order = np.argsort(-values, kind="stable")
    first, second = int(order[0]), int(order[1])
    gap = float(values[first] - values[second])
    ambiguous = gap + 1e-12 < AMBIGUITY_GAP
    return {"possible_ambiguity": ambiguous, "score_gap": round(gap, 6),
            "alternative_issue": str(classes[second]) if ambiguous else "",
            "alternative_score": round(float(values[second]), 6) if ambiguous else None}


def predict_text(text, model, review_rule=None):
    if not isinstance(text, str) or not clean_text(text):
        raise ValueError("Enter a nonempty complaint narrative.")
    if len(text) > 50000:
        raise ValueError("The narrative exceeds 50,000 characters. Shorten it and try again.")
    scores = validate_probabilities(model.predict_proba([text]), model.classes_, 1)[0]
    index = int(scores.argmax())
    flag = bool(review_flags([scores[index]], review_rule)[0]) if review_rule else None
    if review_rule and review_rule.get("mode") == "review_all":
        status = "Every suggestion requires human review; no automatic acceptance rule is enabled."
    elif review_rule:
        status = "Flagged using the validation-selected rule; an unflagged result can still be wrong."
    else:
        status = "No matching review rule loaded; review flag is unavailable."
    return {"predicted_issue": str(model.classes_[index]), "model_score": round(float(scores[index]), 6), "needs_review": flag, "review_status": status, "score_note": "Model scores are not calibrated probabilities of correctness.", **candidate_details(scores, model.classes_)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True)
    args = parser.parse_args()
    model_path = ROOT / "artifacts/model.joblib"
    try:
        if not model_path.exists():
            raise ValueError("The model is missing. Complete stage 2 training first.")
        rule_path = ROOT / "artifacts/review_rule.json"
        rule = json.loads(rule_path.read_text()) if rule_path.exists() else None
        if rule and rule.get("model_sha256") != hashlib.sha256(model_path.read_bytes()).hexdigest():
            raise ValueError("The review rule does not match the current model.")
        result = predict_text(args.text, joblib.load(model_path), rule)
    except ValueError as error:
        parser.exit(2, f"Error: {error}\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
