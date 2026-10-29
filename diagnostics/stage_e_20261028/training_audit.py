"""Record every new fit started through this helper, including warnings/failures."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
from sklearn.exceptions import ConvergenceWarning


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1048576),b""):
            digest.update(block)
    return digest.hexdigest()


def encode(value):
    if callable(value):
        return f"{value.__module__}.{value.__name__}"
    if hasattr(value,"tolist"):
        return value.tolist()
    return str(value)


def write_json(path,payload):
    Path(path).write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=encode)+"\n",encoding="utf-8")


def logged_fit(model,train,validation,*,run_id,data_version,train_path,validation_path,run_dir,ledger_path,metrics_function,seed):
    """Fit train only and persist started/completed/failed events; never overwrite."""
    run_dir,ledger_path=Path(run_dir),Path(ledger_path)
    run_dir.mkdir(parents=True,exist_ok=False)
    record={"run_id":run_id,"status":"started","started_at_utc":datetime.now(timezone.utc).isoformat(),"data_version":data_version,"data_sha256":{"train":sha(train_path),"validation":sha(validation_path)},"rows":{"train":len(train),"validation":len(validation)},"seed":seed,"configuration":{name:{"type":type(step).__name__,"parameters":step.get_params(deep=False)} for name,step in model.steps},"warnings":[],"fit_seconds":None,"n_iter_":None,"validation_metrics":None}

    def event():
        write_json(run_dir/"training_record.json",record)
        ledger_path.parent.mkdir(parents=True,exist_ok=True)
        with ledger_path.open("a",encoding="utf-8") as stream:
            stream.write(json.dumps(record,ensure_ascii=False,default=encode)+"\n")

    event()
    caught=[]
    started=time.monotonic()
    fit_finished=None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model.fit(train.narrative,train.issue)
        fit_finished=time.monotonic()-started
        classifier=model.named_steps["classifier"]
        iterations=classifier.n_iter_
        record.update({"fit_seconds":fit_finished,"n_iter_":encode(iterations),"max_iter":classifier.get_params().get("max_iter"),"optimizer":classifier.get_params().get("solver","liblinear (LinearSVC)"),"tol":classifier.get_params().get("tol")})
        record["warnings"]=[{"category":warning.category.__name__,"message":str(warning.message)} for warning in caught]
        record["convergence_warning"]=any(issubclass(warning.category,ConvergenceWarning) for warning in caught)
        largest=max(iterations) if hasattr(iterations,"__len__") else iterations
        record["iteration_limit_reached"]=int(largest)>=record["max_iter"]
        predicted=model.predict(validation.narrative)
        metrics=metrics_function(validation.issue,predicted,model.classes_.tolist())
        record["validation_metrics"]=metrics
        joblib.dump(model,run_dir/"model.joblib")
        record.update({"model_sha256":sha(run_dir/"model.joblib"),"status":"completed","finished_at_utc":datetime.now(timezone.utc).isoformat(),"elapsed_seconds":time.monotonic()-started})
        event()
        return model,record
    except Exception as error:
        record.update({"status":"failed","fit_seconds":fit_finished if fit_finished is not None else time.monotonic()-started,"warnings":[{"category":w.category.__name__,"message":str(w.message)} for w in caught],"error_type":type(error).__name__,"error":str(error),"finished_at_utc":datetime.now(timezone.utc).isoformat()})
        event()
        raise
