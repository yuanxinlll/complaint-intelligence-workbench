"""Five predeclared validation-only experiments; no historical test inference."""

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.data import clean_text
from src.train import validation_metrics

SEED = 42
DATA = ROOT / "diagnostics/stage_c_20261028/data"
CONFIGURATIONS = [
    {"id": "D01", "name": "Current parameters on v2 data", "analyzer": "word", "ngram_range": [1,2], "min_df": 2, "max_features": 40000, "classifier": "LogisticRegression", "class_weight": "balanced", "C": 1.0, "purpose": "Same-data reproducible reference; distinguish data policy changes from model changes."},
    {"id": "D02", "name": "Remove class balancing", "analyzer": "word", "ngram_range": [1,2], "min_df": 2, "max_features": 40000, "classifier": "LogisticRegression", "class_weight": None, "C": 1.0, "purpose": "Measure majority/minority trade-off of existing balancing."},
    {"id": "D03", "name": "Unigrams only", "analyzer": "word", "ngram_range": [1,1], "min_df": 2, "max_features": 40000, "classifier": "LogisticRegression", "class_weight": "balanced", "C": 1.0, "purpose": "Test whether adjacent-word features actually help compared with single words."},
    {"id": "D04", "name": "Character fragments", "analyzer": "char_wb", "ngram_range": [3,5], "min_df": 3, "max_features": 60000, "classifier": "LogisticRegression", "class_weight": "balanced", "C": 1.0, "purpose": "Test word-boundary character fragments against spelling and wording variation; this changes the representation, not just one parameter."},
    {"id": "D05", "name": "LinearSVC", "analyzer": "word", "ngram_range": [1,2], "min_df": 2, "max_features": 40000, "classifier": "LinearSVC", "class_weight": "balanced", "C": 1.0, "purpose": "Compare the classifier with the same word representation. Decision scores are not probabilities."},
]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")


def log(message):
    line = f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with (OUT / "experiment.log").open("a", encoding="utf-8") as stream:
        stream.write(line+"\n")


def build_model(config):
    vectorizer = TfidfVectorizer(preprocessor=clean_text, lowercase=False, analyzer=config["analyzer"], ngram_range=tuple(config["ngram_range"]), min_df=config["min_df"], max_df=.98, max_features=config["max_features"], sublinear_tf=True, **({"token_pattern": r"(?u)\b\w+\b"} if config["analyzer"]=="word" else {}))
    if config["classifier"] == "LogisticRegression":
        classifier = LogisticRegression(C=config["C"], class_weight=config["class_weight"], max_iter=1000, solver="lbfgs", random_state=SEED)
    else:
        classifier = LinearSVC(C=config["C"], class_weight=config["class_weight"], dual="auto", max_iter=5000, random_state=SEED)
    return Pipeline([("tfidf",vectorizer),("classifier",classifier)])


def main():
    if (OUT / "plan.json").exists():
        raise FileExistsError("A plan/run already exists. Do not silently rerun or add experiments.")
    protected = []
    for folder in ["src", "tests", "artifacts", "data", "reports", "diagnostics/stage_a_20261028", "diagnostics/stage_b_20261028", "diagnostics/stage_c_20261028"]:
        protected += [p for p in (ROOT/folder).rglob("*") if p.is_file() and not any(part in {"__pycache__","plot_cache","matplotlib_cache"} for part in p.parts)]
    protected += [ROOT/p for p in ["app.py","README.md","PROJECT_STATUS.md","STATUS.md"]]
    fingerprints = {str(p.relative_to(ROOT)):sha(p) for p in protected}
    data_manifest = json.loads((DATA.parent / "manifest.json").read_text())
    for split in ["train","validation"]:
        if sha(DATA/f"{split}.csv") != data_manifest["data_files"][f"data/{split}.csv"]["sha256"]:
            raise ValueError(f"Data version mismatch: {split}")
    train = pd.read_csv(DATA/"train.csv",dtype=str,keep_default_na=False)
    validation = pd.read_csv(DATA/"validation.csv",dtype=str,keep_default_na=False)
    labels = sorted(train.issue.unique().tolist())
    majority = train.issue.value_counts().index[0]
    minority = sorted(train.issue.value_counts().loc[lambda counts:counts<200].index.tolist())
    plan = {
        "created_at_utc":datetime.now(timezone.utc).isoformat(), "max_experiments":5,"configurations":CONFIGURATIONS,
        "seed":SEED,"data_version":data_manifest["version"],"train_rows":len(train),"validation_rows":len(validation),
        "data_sha256":{split:sha(DATA/f"{split}.csv") for split in ["train","validation"]},
        "selection_policy":{"primary":"validation Macro-F1", "majority_class":majority,"minority_classes":minority,"minority_definition":"fewer than 200 training examples, fixed before outcomes", "minimum_macro_f1_gain_over_D01":.01,"maximum_majority_f1_drop":.03,"maximum_mean_minority_f1_drop":.02,"tie_break":"highest Macro-F1, then shorter fit time, then experiment ID", "fallback":"Keep D01 as same-data reference if no alternative meets all gates; do not force a change to the deployed old model.","caveat":"Gates are operational comparison criteria, not statistical significance tests or reliability guarantees."},
        "test_policy":"Do not predict on or inspect historical test results for experiment/model selection. Existing test files only hashed for preservation.",
        "review_policy":"No threshold selection, calibration, or old 0.45 threshold reuse in this stage.",
        "python":platform.python_version(),"dependencies":{name:importlib.metadata.version(name) for name in ["scikit-learn","pandas","numpy","scipy","joblib"]},
        "experiment_script_sha256":sha(Path(__file__)),"original_files_sha256":fingerprints,
    }
    write_json(OUT/"plan.json",plan)
    log("Plan frozen: five trials, same6333 training and1428 validation records; no historical test.")
    log("Gate: Macro-F1 gain>=.01, majority F1 drop<=.03, minority mean F1 drop<=.02.")
    comparisons=[]
    for config in CONFIGURATIONS:
        run=OUT/"runs"/config["id"]
        run.mkdir(parents=True)
        write_json(run/"configuration.json",config)
        log(f"[{config['id']}/5] Start {config['name']}")
        start=time.monotonic()
        try:
            model=build_model(config)
            with warnings.catch_warnings():
                warnings.simplefilter("error",ConvergenceWarning)
                model.fit(train.narrative,train.issue)
            fit_seconds=time.monotonic()-start
            predict_start=time.monotonic()
            predicted=model.predict(validation.narrative)
            metrics=validation_metrics(validation.issue,predicted,labels)
            if config["classifier"]=="LinearSVC":
                scores=model.decision_function(validation.narrative)
                score_kind="uncalibrated decision_function; NOT probabilities"
            else:
                scores=model.predict_proba(validation.narrative)
                score_kind="uncalibrated predict_proba class output; NOT correctness probability"
            ordering=np.argsort(-scores,axis=1,kind="stable")
            output=validation[["complaint_id","received_date","issue"]].copy()
            output["predicted_issue"]=predicted
            output["top1_score"]=scores[np.arange(len(validation)),ordering[:,0]]
            output["second_issue"]=model.classes_[ordering[:,1]]
            output["top2_score"]=scores[np.arange(len(validation)),ordering[:,1]]
            output["score_gap"]=output.top1_score-output.top2_score
            output.to_csv(run/"validation_predictions.csv",index=False)
            prediction_seconds=time.monotonic()-predict_start
            joblib.dump(model,run/"model.joblib")
            loaded=joblib.load(run/"model.joblib")
            assert np.array_equal(loaded.predict(validation.narrative.head(8)),predicted[:8])
            write_json(run/"metrics.json",{"split":"validation","metrics":metrics,"score_kind":score_kind,"threshold":None,"test_evaluated":False})
            row={"id":config["id"],"name":config["name"],"status":"success","macro_f1":metrics["macro_f1"],"accuracy":metrics["accuracy"],"majority_f1":metrics["per_class"][majority]["f1-score"],"minority_mean_f1":float(np.mean([metrics["per_class"][label]["f1-score"] for label in minority])),"fit_seconds":fit_seconds,"prediction_seconds":prediction_seconds,"features":len(model.named_steps["tfidf"].vocabulary_),"model_bytes":(run/"model.joblib").stat().st_size,"model_sha256":sha(run/"model.joblib"),"score_kind":score_kind,"reload_check":True}
            for index,label in enumerate(labels):
                for metric in ["precision","recall","f1-score","support"]:
                    row[f"class_{index}_{metric}"]=metrics["per_class"][label][metric]
            write_json(run/"run.json",row)
            comparisons.append(row)
            log(f"{config['id']} complete: Macro-F1={row['macro_f1']:.4f}, Accuracy={row['accuracy']:.4f}, minority mean F1={row['minority_mean_f1']:.4f}, fit {fit_seconds:.2f} seconds.")
        except Exception as error:
            failure={"id":config["id"],"name":config["name"],"status":"failed","error_type":type(error).__name__,"error":str(error),"elapsed_seconds":time.monotonic()-start}
            comparisons.append(failure)
            write_json(run/"failure.json",failure)
            log(f"{config['id']} failed and recorded; no changed-config retry: {type(error).__name__}")
        pd.DataFrame(comparisons).to_csv(OUT/"comparison.csv",index=False)
    successful=[row for row in comparisons if row["status"]=="success"]
    baseline=next((row for row in successful if row["id"]=="D01"),None)
    if baseline is None:
        selection={"recommended_candidate":None,"reason":"D01 failed; no justified same-data baseline comparison."}
    else:
        eligible=[row for row in successful if row["id"]!="D01" and row["macro_f1"]>=baseline["macro_f1"]+.01 and row["majority_f1"]>=baseline["majority_f1"]-.03 and row["minority_mean_f1"]>=baseline["minority_mean_f1"]-.02]
        selected=sorted(eligible,key=lambda row:(-row["macro_f1"],row["fit_seconds"],row["id"]))[0] if eligible else baseline
        selection={"recommended_candidate":selected["id"],"eligible_alternatives":[row["id"] for row in eligible],"baseline_id":"D01","reason":"Highest Macro-F1 among alternatives passing frozen gates." if eligible else "No alternative passed all frozen gates; retain the same-data reference, without switching the deployed model.","candidate_model_sha256":selected["model_sha256"]}
    changed=[name for name,value in fingerprints.items() if sha(ROOT/name)!=value]
    assert not changed
    selection.update({"selected_at_utc":datetime.now(timezone.utc).isoformat(),"labels":labels,"test_used_for_selection":False,"review_rule_selected":False,"app_switched":False,"protected_files_unchanged":True,"changed_original_files":changed,"attempted_experiments":len(comparisons),"successful_experiments":len(successful)})
    write_json(OUT/"selection.json",selection)
    log(f"Stage D complete; candidate={selection['recommended_candidate']}. No app switch or next-stage execution.")


if __name__=="__main__":
    main()
