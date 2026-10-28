# Complaint Intelligence Workbench

A local complaint classification prototype for the Solix AI Hackathon 2026, Data + AI track. Turn English debt collection narratives into issue categories, model scores, and human review flags without external model APIs.

## Features

- Single-complaint and batch CSV classification across seven issue categories.
- Filters, category distribution, and CSV export.
- TF-IDF + logistic regression with chronological evaluation and a validation-selected review rule.
- Local processing with no API key required.

## Results

CFPB November–December 2024 data: 11,706 labeled narratives, reduced to 9,177 after normalized deduplication. Training: 6,351; validation: 1,428; reserved test: 1,398.

| Model | Validation accuracy | Validation Macro-F1 | Test accuracy | Test Macro-F1 |
|---|---:|---:|---:|---:|
| Majority-class baseline | 44.47% | 0.0879 | 46.57% | 0.0908 |
| TF-IDF + logistic regression | 52.59% | 0.3975 | 48.50% | 0.3888 |

The frozen review threshold is 0.45. On the test set, 325 records (23.25%) were unflagged, with 70.77% accuracy and 95 errors. Scores are not calibrated correctness probabilities; unflagged predictions still require judgment. See [evaluation results](reports/SHARED_RESULTS.md).

## Setup

The recorded run used Python 3.14.6. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-evaluation.txt
python -m unittest discover -s tests -v
```

On Windows, activate with `.venv\Scripts\activate`. Tests use synthetic fixtures and require no complaint data.

### Prepare data and model

Download **CCDB Export November 2024 through December 2024** from the [CFPB historical narrative archive](https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/). Use the narrative archive, not structured records without complaint text.

The original file used for the recorded run has SHA-256:

```text
2ef7d563bf6283a1282bcfba2d99d72f9bfa4a4617524682de72a9b3fa47bed0
```

```bash
python src/inspect_archive.py /path/to/downloaded_archive.csv
python -u -m src.train
python -u -m src.evaluate
```

The scripts preserve the source archive, create a debt collection subset, train the model, freeze the review rule, and evaluate the test set. Existing data and runs are protected from overwrite; use a fresh project copy to reproduce a run. Logs and model outputs are saved in `artifacts/`.

### Launch the workbench

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Open http://127.0.0.1:8501. The **How it works** page explains the pipeline. Classification loads the saved model without retraining. A fresh clone requires the data preparation, training, and evaluation steps above.

CSV input requires `narrative` (or `Consumer complaint narrative`); `complaint_id` and `issue` are optional. Supplied labels appear as `original_issue` and never enter the model. Limits: UTF-8, 5 MB, 500 rows, 50,000 characters per narrative, and 2 million narrative characters total. Invalid rows retain error messages; duplicate rows remain in batch statistics. The three demo examples are synthetic.

CLI example (synthetic input):

```bash
python -m src.predict --text 'I asked for written proof of this debt but received no notice.'
```

Returns `predicted_issue`, `model_score`, and `needs_review`. Without a rule, the review flag is `null`; a mismatched rule is rejected. Only load model files you trained or trust.

## Project layout

| File | Purpose |
|---|---|
| `app.py` | Streamlit workbench |
| `src/inspect_archive.py` | Archive inspection and subset extraction |
| `src/data.py` | Normalization, deduplication, and chronological splits |
| `src/train.py` | Baseline and text-model training |
| `src/review.py`, `src/evaluate.py` | Review rule selection and reserved-test evaluation |
| `src/predict.py`, `src/interface.py` | Single and batch inference |
| `tests/` | Core logic and CSV validation tests |

## Scope and data handling

This prototype supports review, not automated financial decisions. Consumer-selected labels can be noisy; the two-month public sample is imbalanced and may contain similar templates. Single-label predictions cannot capture every concern in a complaint, and scores do not reliably detect unknown categories.

Raw data, models, and predictions containing complaint text are excluded from the repository. No Solix product integration or official endorsement is claimed. No project license has been granted; do not assume redistribution rights.
