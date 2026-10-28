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


def candidate_details(scores, classes):
    values = np.asarray(scores, dtype=float)
    order = np.argsort(-values, kind="stable")
    first, second = int(order[0]), int(order[1])
    gap = float(values[first] - values[second])
    ambiguous = gap + 1e-12 < AMBIGUITY_GAP
    return {"possible_ambiguity": ambiguous, "score_gap": round(gap, 6),
            "alternative_issue": str(classes[second]) if ambiguous else "",
            "alternative_score": round(float(values[second]), 6) if ambiguous else None}


def predict_text(text, model, review_rule=None):
    if not clean_text(text):
        raise ValueError("Enter a nonempty complaint narrative.")
    if len(text) > 50000:
        raise ValueError("The narrative exceeds 50,000 characters. Shorten it and try again.")
    scores = model.predict_proba([text])[0]
    index = int(scores.argmax())
    flag = bool(review_flags([scores[index]], review_rule)[0]) if review_rule else None
    status = "Flagged using the validation-selected rule; an unflagged result can still be wrong." if review_rule else "No matching review rule loaded; review flag is unavailable."
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
