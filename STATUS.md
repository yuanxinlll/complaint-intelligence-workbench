# Project status — October 28, 2026

- Main local application defaults to FT01 DistilBERT epoch 3, selected by development Macro-F1 .4629 (Accuracy52.24%).
- Model selector retains DATA100 (.4409/53.08%) and original v1 (.3975/52.59%). No model artifacts were overwritten.
- Single/CSV/local EML review, ambiguity alternatives, export, English learning guide and model-results page are integrated. Export identifies model version; long-text truncation is visible.
- FT01 improves two small-class F1 scores and decreases the other five versus DATA100. It is slower and has lower overall Accuracy. Same phone sensitivity tests have not yet been run on FT01.
- All results remain development evidence on a reused validation set, not independent final generalization evidence.
- Original run's cache-only terminal assertion is preserved and independently audited. Three-epoch results and checkpoint are complete.
- Model loading / saved-output consistency and synthetic single, email, batch, invalid input, long-text and export checks passed; no new training.
- Weights, complaint records, per-record predictions, review journals and credentials remain local and are not committed.
- Final human-confirmation storage, mailbox connection and online retraining are not implemented. No deployment or automatic submission.
