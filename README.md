# Complaint Intelligence Workbench

A local English debt-collection classification assistant for human review. Paste a narrative, import a local `.eml` message and check its subject/body, or classify a small UTF-8 CSV. Every suggestion needs a human decision. This prototype does not store final confirmation decisions.

## Online demo setup

The lightweight hosted entrypoint is `cloud_demo/streamlit_app.py`, using DATA100 (53.08% development Accuracy, Macro-F1 0.4409). The one trusted 3.86 MB DATA100 artifact is included with explicit owner authorization. No complaint records or FT01 weights are shared. Inputs to a hosted app leave the browser for its server: use synthetic text only. [Deployment instructions and current status](cloud_demo/README.md). A Live Demo link will appear here after deployment is verified.

## Active model and rollback

The default local app now uses **FT01, DistilBERT epoch 3**, selected by development Macro-F1. The model selector retains **DATA100** (fast TF-IDF + balanced logistic regression) and original **v1** for comparison/rollback. No online training, external AI API or automatic downloading is used.

| Version | Training records | Accuracy | Macro-F1 |
|---|---:|---:|---:|
| FT01 epoch 3 — default | 19,223 | 52.24% | **0.4629** |
| DATA100 — fast rollback | 19,223 | **53.08%** | 0.4409 |
| v1 — original reference | 6,351 | 52.59% | 0.3975 |
| AUG01 — research only | 19,241 | 53.29% | 0.4433 |

All results use the repeatedly viewed 1,428-record development validation. There is no fresh independent final test. FT01 improves F1 for electronic communications and third-party disclosure threats, while the other five class F1 scores decrease versus DATA100. It costs more to run and has lower overall Accuracy and Top-2 inclusion. AUG01's tiny gain did not meet its predeclared recommendation gate. Do not describe any model as universally best.

## Features and limits

- One suggested category; an alternative appears for a top-two score gap below 0.10. This is an unvalidated display heuristic, not calibrated correctness.
- Original text remains available. Scores and details are expandable.
- Local `.eml` import extracts subject/body for user review. Attachments are ignored, no mailbox is connected, and remote images/links are not fetched. Email-domain accuracy is unmeasured.
- CSV classification, filters, batch statistics and export of unconfirmed suggestions with model version and truncation indicators. Invalid rows remain visible.
- English learning guide and recorded model-results page.

FT01's input cap is 512 tokens including two special tokens. Over 510 content tokens, it uses the first and last 255 and omits the middle; the app warns when this happens. Original text is not overwritten. This can omit important facts. No old model's acceptance threshold is transferred; all records require review.

Phone-word sensitivity was observed in synthetic word-model probes. Simple downweighting and fallback rules did not improve full validation and are not part of the default app. **FT01 has not yet undergone the same keyword-robustness probe.** No claim that the issue is fixed.

## Local setup

Complaint records, FT01/v1 weights, review answers, environments and secrets are excluded. The explicit exception is the trusted DATA100 artifact for `cloud_demo/streamlit_app.py`. A fresh clone can run that demo after installing dependencies; `app.py` still requires its separate local model artifacts. The app explains missing files rather than downloading or silently choosing a different model.

The recorded environment is macOS Apple Silicon, Python 3.14.6. The pinned semantic runtime was already installed in an isolated environment; collaborators should review package/platform compatibility and download size before installing on their own machines.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-semantic.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats false
```

Windows activation: `.venv\Scripts\activate`. Installing dependencies is a separate network action; these instructions do not run automatically. No Hugging Face token or paid inference API is needed for local inference with trusted files already present.

## Model artifacts and the authorized cloud exception

`config/model_profiles.json` records exact version paths, checksums and development metrics. Expected local assets:

- `artifacts/versions/ciw-distilbert-ft01-epoch3/`: model.safetensors, config.json, tokenizer.json, tokenizer_config.json.
- `artifacts/versions/ciw-lr-data100/model.joblib`.
- `artifacts/model.joblib` for original v1.

FT01 artifacts are about 268.56 MB; DATA100 is about 3.86 MB. Obtain/recreate trusted artifacts separately; **do not commit additional artifacts or bypass checksum verification**. The exact `cloud_demo/model.joblib` is the separately authorized DATA100 hosting exception. Reproducing training can produce different bytes and requires a new version entry/checksum rather than pretending it is this checkpoint. Joblib can execute code: never load untrusted weights.

The original word-model workflow uses the official historical CFPB narrative archive, `python -m src.inspect_archive /path/to/archive.csv`, then `python -m src.train`. The historical evaluation module reproduces an already-viewed holdout workflow; do not rerun it to tune models or claim a new test. These original commands do not automatically recreate DATA100 or FT01. Later research sources require their separately prepared, versioned local inputs; see [research summary](docs/RESEARCH_SUMMARY.md).

## Training and verification

FT01: one fixed CPU run, three epochs, seed 42, AdamW 2e-5, weight decay 0.01, float32, effective batch 16 via single-record accumulation. Training-only inverse-frequency weights are normalized over each effective batch; an equivalence check prevents single-record weighted-mean cancellation. TF-IDF baselines fit features only on training data.

Epochs: Macro-F1 .4330, .4239, .4629; Accuracy 47.55%, 48.46%, 52.24%. Total workflow about 2h45m38s; peak process memory about 2.47 GB. Model selection was by Macro-F1, not by the last epoch simply being last. Repeated validation and checkpoint selection remain sources of optimism.

The original run ended with a preservation assertion on two Python cache files after all results and the checkpoint were saved. A separate audit verified source/data/old models unchanged and recomputed metrics from saved predictions. Original failure logs were retained; this was a cache-only audit warning, not a hidden training failure. Integration additionally reloaded the model and checked saved outputs on three validation records (verification only, not selection).

The read-only progress source is in `research/discrimination_20261028/finetune_run/`. It needs local status files; GitHub does not contain a live training state. Use the main app model selector to roll back without overwriting original files.

## Collaboration and presentation

Use English for repository code/documentation. [CONTRIBUTING.md](CONTRIBUTING.md) covers data and evaluation isolation. [DEMO.md](DEMO.md), [presentation/DEVPOST_DRAFT.md](presentation/DEVPOST_DRAFT.md) and the pitch are presentation materials, not proof of submission. Research failures and limitations are preserved in `docs/`. The code repository remains private. The owner authorized preparing a DATA100 hosted demo; current deployment status is in `cloud_demo/README.md`. No automatic competition submission occurs.
