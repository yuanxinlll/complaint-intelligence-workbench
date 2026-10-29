"""Keyword sensitivity probe for the Complaint Intelligence Workbench.
Run locally from the repo root (synthetic texts only):
  python keyword_sensitivity.py --model artifacts/model.joblib --texts probe_texts.txt
Only load joblib files you trained yourself.
"""
import argparse, csv, re
from collections import defaultdict
import joblib

INSERT_WORDS = ["call", "phone", "calls", "letter", "mail", "written", "email", "lawsuit", "court",
                "sue", "attorney", "credit report", "not mine", "never owed", "harass",
                "threatened", "validation", "cease", "work", "wage"]

def split(obj):
    if isinstance(obj, dict):
        obj = obj.get("pipeline") or obj.get("model") or next(iter(obj.values()))
    return obj

def predict(model, texts):
    out = []
    for t in texts:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba([t])[0]; idx = p.argsort()[::-1]
            cls = model.classes_
            out.append((cls[idx[0]], float(p[idx[0]] - p[idx[1]])))
        else:
            out.append((model.predict([t])[0], 0.0))
    return out

def top_terms(model, k=15):
    steps = getattr(model, "steps", None)
    if not steps: return
    vec, clf = steps[0][1], steps[-1][1]
    if not hasattr(clf, "coef_") or not hasattr(vec, "get_feature_names_out"): return
    names = vec.get_feature_names_out()
    with open("top_terms_per_class.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["class", "rank", "term", "weight"])
        coefs = clf.coef_ if clf.coef_.shape[0] > 1 else [-clf.coef_[0], clf.coef_[0]]
        for c, row in zip(clf.classes_, coefs):
            for r, i in enumerate(row.argsort()[::-1][:k], 1):
                w.writerow([c, r, names[i], round(float(row[i]), 4)])
    print("wrote top_terms_per_class.csv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/model.joblib")
    ap.add_argument("--texts", default="probe_texts.txt")
    a = ap.parse_args()
    model = split(joblib.load(a.model))
    texts = [l.strip() for l in open(a.texts, encoding="utf-8") if l.strip()]
    base = predict(model, texts)
    top_terms(model)

    # 1) Occlusion: delete each word, count prediction flips
    stats = defaultdict(lambda: [0, 0])
    for t, (b, _) in zip(texts, base):
        words = t.split()
        for i, wd in enumerate(words):
            key = re.sub(r"\W+", "", wd.lower())
            if not key: continue
            new = " ".join(words[:i] + words[i + 1:])
            stats[key][1] += 1
            if predict(model, [new])[0][0] != b: stats[key][0] += 1
    with open("occlusion_flips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["word", "flips", "tested"])
        for k, (fl, n) in sorted(stats.items(), key=lambda x: (-x[1][0], x[0])):
            if fl: w.writerow([k, fl, n])
    print("wrote occlusion_flips.csv (words whose removal changed a prediction)")

    # 2) Insertion: append a keyword to every probe text
    with open("insertion_flips.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["keyword", "flips", "texts", "new_labels"])
        for kw in INSERT_WORDS:
            res = predict(model, [f"{t} {kw}." for t in texts])
            fl = sum(r[0] != b[0] for r, b in zip(res, base))
            w.writerow([kw, fl, len(texts), "|".join(sorted({str(r[0]) for r in res}))])
    print("wrote insertion_flips.csv")

if __name__ == "__main__":
    main()
