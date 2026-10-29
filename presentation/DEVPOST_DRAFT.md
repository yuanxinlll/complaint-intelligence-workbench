# Devpost project description — English draft

**Complaint Intelligence Workbench** — local complaint classification suggestions with human review.

## Inspiration

English debt-collection complaints mix debt ownership, missing notices, contact behavior and legal threats. We built a small working assistant that preserves the original text and makes uncertain suggestions reviewable.

## What it does

Paste a narrative, import a UTF-8 CSV, or review subject/body extracted from a local EML email. The app suggests one of seven categories, may show an alternative, summarizes batches and exports unconfirmed suggestions with model version. No mailbox connection or saved final human-confirmation workflow is claimed.

## How we built it

Historical CFPB narratives, chronological exact-normalized deduplication, training-only feature fitting, explicit training versions and limited recorded experiments. Original TF-IDF/logistic regression remains selectable. The default completed DistilBERT model was fine-tuned in one fixed CPU run and selected by development Macro-F1. Long-text truncation is disclosed.

## Evidence

FT01 epoch3: Macro-F1 .4629, Accuracy 52.24%; DATA100 .4409/53.08%. FT01 improves two small-class F1 scores but lowers the other five and costs more to run. All metrics are from the repeatedly used1428-record development validation; no fresh independent final test or enterprise efficiency result is claimed.

## Challenges and learning

We found synthetic keyword/repetition sensitivity, tried simple attenuation and a fallback rule, and retained the originals when full-validation results did not improve. On five texts, call changed 2 of 5 predictions and today changed 1 of 5, insufficient to isolate a unique keyword effect. A frozen semantic encoder and equal soft voting did not improve the best word model. A reviewed-synthetic augmentation gain was too small to recommend. FT01 keyword robustness is not yet tested.

## Next steps

Independent reviewer agreement, genuinely unused evaluation data, stored human confirmation and carefully evaluated retrieval. Scores are uncalibrated; all suggestions need human judgment. No automatic online learning, deployment or submission.
