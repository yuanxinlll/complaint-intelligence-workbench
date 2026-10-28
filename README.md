# Complaint Intelligence Workbench

A local English debt-collection complaint classification assistant for the Solix AI Hackathon 2026 Data + AI track. The model suggests a category; a person must read the narrative and confirm the decision. This is a research prototype, not an autonomous complaint-processing system.

## What is implemented

- Single-narrative and UTF-8 CSV classification across seven CFPB Issue categories.
- One suggested category; a possible-ambiguity hint and alternative when the top-two score gap is below 0.10.
- Narrative inspection, batch filtering, category counts and export of unconfirmed suggestions.
- English project overview and learning guide.
- A separate local progress monitor for the current fine-tuning experiment.

The 0.10 gap is a display heuristic, not a validated guarantee. Scores are uncalibrated; a high score or an absent warning does not authorize automatic adoption. Human confirmation storage is not implemented.

## Model status and honest results

All rows below use the same 1,428-record development validation set, which has been repeatedly viewed. Training versions differ as shown. There is no new independent final test.

| Model / experiment | Training records | Accuracy | Macro-F1 | Status |
|---|---:|---:|---:|---|
| Active application v1, balanced word/bigram TF-IDF + LR | 6,351 | 52.59% | 0.3975 | Still powers the app |
| D01 fixed baseline after training-only quarantine | 6,333 | 51.96% | 0.3931 | Reused baseline |
| D03 unigram TF-IDF + LR | 6,333 | 50.28% | 0.4017 | Historical best word-feature configuration |
| DATA50, fixed D01 + half of the eligible additions | 12,775 | 53.15% | 0.4217 | Development candidate |
| DATA100, fixed D01 + all eligible additions | 19,223 | 53.08% | 0.4409 | Best development candidate; not integrated |
| SEM01, frozen MiniLM + LR | 6,333 | 44.26% | 0.3834 | Not adopted |
| ENS01, equal DATA100 / SEM01 soft vote | Mixed versions | 51.47% | 0.4384 | No improvement; not adopted |
| FT01, task-specific DistilBERT fine-tuning | 19,223 | Pending | Pending | Running at the time of this update |

Increasing historical data improved development Macro-F1, but DATA100 slightly reduced written-notification F1 relative to D01. Small categories remain weak. The ensemble corrected 68 DATA100 errors but lost 91 previously correct predictions. Failed experiments are retained in the research summary.

The historical 1,398-record holdout has already been viewed. It is not used for new model selection and must not be described as a fresh independent final test. Template similarity and label ambiguity limit conclusions. See [research summary](docs/RESEARCH_SUMMARY.md) and [model card](docs/MODEL_CARD.md).

## Run the application

This repository contains code and aggregate documentation only. Complaint records, source archives, trained weights, per-record predictions, review forms and credentials are deliberately excluded. A fresh clone cannot classify complaints until the documented local artifacts have been supplied or recreated.

The original application was validated with Python 3.14.6 and the pinned dependencies. Installing packages requires network access and is a collaborator action, not part of opening this README.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-evaluation.txt
python -m unittest discover -s tests -v
python -m streamlit run app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
```

Windows activation: `.venv\Scripts\activate`. Open http://127.0.0.1:8501 locally. `?view=learn` opens the English learning guide. Without local artifacts, the app explains what is missing; weights are not downloaded automatically.

## Local data and original pipeline

Acquire official CFPB historical narrative archives separately, subject to the source's current terms: https://www.consumerfinance.gov/foia-requests/foia-electronic-reading-room/cfpb-consumer-complaint-database-narratives-archive/

The original workflow is `python -m src.inspect_archive /path/to/the/archive.csv`, then `python -m src.train`, then the historical `python -m src.evaluate`, using the paths specified by those modules. Read their CLI/options and input requirements first. These commands generate local data/model/evaluation artifacts and are not automatically run. The evaluation script reproduces a historical workflow; it does not create a fresh independent test. Do not rerun it to select new models.

The historical baseline scripts do not recreate the later DATA100 research experiment. That requires separately authorized archives, the training-only quarantine and frozen augmentation lists described in the research summary. Do not simply merge all records or randomly resplit old training data and call it independent testing.

App assets: `artifacts/model.joblib`, `artifacts/review_rule.json` and metadata referenced by `src/interface.py`. Never load joblib artifacts from an untrusted source. Input CSV requires `narrative`; optional `complaint_id` and `issue` are not prediction features. Batch limits: 5 MB / 500 rows, 50,000 characters per narrative and 2,000,000 narrative characters total. Invalid rows remain visible.

## Fine-tuning and progress monitor

The experiment source is included at `research/discrimination_20261028/finetune_run/`. It requires local DATA100 data, an explicitly downloaded DistilBERT snapshot, validation files and a compatible isolated PyTorch environment. These inputs and environments are not committed. Do not run it until resource and data checks are complete. It refuses to overwrite an existing run.

Fine-tuning is one fixed CPU run, maximum three epochs and eight hours / 6 GB process RSS, with one separate longest-input safety update whose parameters are discarded. Validation Macro-F1 selects a checkpoint; results are development evidence. The training run does not update the application model.

```bash
python -m streamlit run research/discrimination_20261028/finetune_run/progress_app.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats false
```

The monitor only reads local `state.json` and `training.log`; it does not start training or ship live results in GitHub.

## Collaboration

Keep the repository private. Use a feature branch and pull request for code changes. Do not commit real complaint text, identifiers, model weights, local environments or secrets. Tests use synthetic fixtures. No external AI API, mailbox connection, automatic online learning or cloud deployment is implemented. See [contribution guidance](CONTRIBUTING.md), [presentation draft](presentation/DEVPOST_DRAFT.md), and [project status](STATUS.md).
