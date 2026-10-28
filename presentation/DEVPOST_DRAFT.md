# Devpost project description — English draft

Project: **Complaint Intelligence Workbench**

Tagline: **Local complaint classification suggestions that keep human judgment visible.**

Track: **Data + AI (Enterprise AI)**

## Inspiration

Customer complaints rarely arrive as clean, single-issue examples. They mix disputed debts, missing notices, communication problems, and legal threats. We wanted a small working tool that offers a useful starting point without disguising uncertain predictions as final decisions.

## What it does

Our local Streamlit prototype accepts an English debt collection narrative or a UTF-8 CSV and suggests one of seven issue categories. It preserves the narrative for inspection, can reveal an alternative when class scores are close, summarizes batch suggestions, and exports unconfirmed results. Every suggestion needs a human decision; saving that decision is future work.

## How we built it

We used a CFPB historical archive covering November–December 2024. We selected debt collection records with text, normalized exact-text keys, prevented exact text from crossing chronological splits, and fitted TF-IDF only on training narratives. A balanced logistic regression pipeline powers the current app. Python, pandas, NumPy, scikit-learn, joblib, and Streamlit form the application stack. Classification uses no external model API.

## Challenges

Consumer-selected labels can overlap, templates can resemble unrelated complaints, and small categories have few examples. In representative cases, text did not identify a communication channel or expressed multiple reasonable issues. We retained original labels and did not automatically relabel complaints from model predictions.

## Evidence and limitations

The active v1 model's development validation Accuracy is 52.59%, with Macro-F1 0.3975 on 1,428 records. Validation has been repeatedly used, and historical holdout results have been seen; we do not claim a new independent final test. The separate D03 experiment reached Macro-F1 0.4017 with a changed training version. Its 69.47% Top-2 inclusion rate is not Top-1 accuracy or a score for the current app. A frozen semantic encoder underperformed and was not adopted.

The displayed ambiguity hint uses a fixed score-gap heuristic, not a calibrated correctness probability. Reviewers must still read the narrative, including when no hint appears.

## Accomplishments and learning

We built a working local path from messy historical data to reviewable suggestions, preserved model and data versions, and recorded unsuccessful experiments. We learned that a more complex representation is not automatically better and that missing context cannot be repaired by a confidence score.

## What's next

Subject to separate approval: capture human confirmation without overwriting source labels, clarify ambiguous category boundaries, and test independently sourced earlier training records. Operational time savings and real-world generalization remain unmeasured.

---

Organizer-facing fields to complete manually: team member names and registration, actual project link/reviewer access, and final submission. The existing private GitHub copy predates these local edits; do not describe it as synchronized. Do not publish complaint data, machine-specific files, or model artifacts automatically. This is a local prototype, not a hosted service.

## Research update — October 28, 2026

The best offline candidate DATA100 reached development Macro-F1 .4409 and Accuracy53.08% after fixed-configuration historical-data expansion. An equal soft-vote ensemble did not improve it. A task-specific DistilBERT run is ongoing; no completed result is claimed. The active demo still uses v1. No new independent test or automatic model replacement is claimed.
