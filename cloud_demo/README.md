# Hosted synthetic-data demonstration

This entrypoint runs the **DATA100 TF-IDF + balanced logistic regression** model. The separate local app still defaults to FT01 DistilBERT. This is an explicit demo choice for lower startup and memory cost, not a new experiment or a claim that DATA100 is best on every metric.

DATA100 uses 19,223 historical training complaints. On 1,428 repeatedly reused validation records, Accuracy is 53.08% and Macro-F1 is 0.4409. These are development results. No fresh independent final test or email-domain accuracy is claimed.

**Use synthetic examples only.** Inputs and uploaded files travel to the hosting server. The application does not intentionally write submissions to disk, train on them, or call an external AI API. This does not establish a hosting-provider retention guarantee. Never enter real complaint narratives, account numbers, personal information or private email.

Every suggestion needs human review. Scores are uncalibrated. A second candidate may appear when the leading score gap is below 0.10, an unvalidated display heuristic. No prompt does not prove the complaint is unambiguous. Final human decisions are not stored.

## Streamlit Community Cloud deployment

- Repository: `yuanxinlll/complaint-intelligence-workbench`
- Branch: `main`
- Entrypoint: `cloud_demo/streamlit_app.py`
- Python: select **3.13** in Advanced settings
- Dependencies: `cloud_demo/requirements.txt`, the six recorded word-model runtime pins
- No secrets, API key, paid inference service, GPU or new model download
- If deploying from a private repository, review app sharing separately: a private app will not be accessible to judges without permission.

The included 3,863,492-byte `model.joblib` is the user-authorized deployment exception. Its SHA256 is `e9fb35f684d264855799dbfac4b398e989ddb7790bd50ab90473c30d2c313bc2`. The loader verifies it before trusted joblib loading. No raw complaint data, per-record predictions, review journals or FT01 weights are included. No retraining occurs at startup.

The cloud dependencies install on the hosting server during deployment. The same package versions passed local checks in Python 3.14.6; Linux/Python 3.13 startup must still be verified from the deployment logs. Free hosting can sleep or restart. Keep local screenshots/video as a presentation backup.

## Local smoke check

From the repository root, in an already prepared word-model environment:

```bash
python -m streamlit run cloud_demo/streamlit_app.py
```

Running `app.py` directly keeps the local model registry and requires separately obtained local model artifacts.

## Deployment status

Live demo: [https://complaint-intelligence-workbench-yzgqmjj3iz9diac9jwjxns.streamlit.app/](https://complaint-intelligence-workbench-yzgqmjj3iz9diac9jwjxns.streamlit.app/).

Verified on October 28, 2026 from a browser without a Streamlit login prompt: DATA100 loaded, a synthetic written-notice complaint produced Written notification about debt, and all three built-in batch examples classified with zero input errors. The model version and hosted-data warning were visible. Local input/parser/export checks also passed. These smoke checks do not measure model accuracy or establish continuous availability. Public hosting and source-repository visibility are separate settings.
