# Integration and rollback

Main app defaults to FT01 by recorded development Macro-F1. `config/model_profiles.json` contains local relative paths and exact checksums. `src/runtime_models.py` loads trusted local files only, applies training-identical512-token head-tail processing, uses CPU/eval/inference mode and serializes access to the shared cached semantic model. No external API or automatic download.

The UI model selector can return to DATA100 or v1. It invalidates previous batch/email results when model versions change. Versioned original weights are never overwritten. All selected models use review-all in the canonical app; old probability thresholds are not transferred. Existing historical preview pages remain local reference artifacts.

Exports record `model_version`, `confirmation_status=unconfirmed` and `input_truncated`; invalid rows remain. Reserved output columns cannot be silently supplied as new input. Long text shows a warning. A model change does not create a human-confirmed label.

Verification:17 main synthetic unit tests,11 standalone review-tool tests, semantic reload on3 existing validation records with max score difference below1.6e-7, and synthetic single/email/batch/invalid/512-token/export/rollback checks. New inference was for integration verification only; no retraining or score-based configuration selection. Fresh-clone no-model behavior is tested separately. Actual artifacts, complaints and review answers remain local.

Reproduce on another computer only after supplying trusted matching artifacts and compatible dependencies. A missing semantic environment or artifact produces an explicit error, never a silent fallback. For deployment or new model selection, obtain separate approval and preserve this version.
