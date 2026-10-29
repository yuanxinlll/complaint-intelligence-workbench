# Code review and regression checks — October 28, 2026

## Scope

Reviewed the current GitHub `main` source, including the application, model registry, inference helpers, CSV and email parsing, original data/train/evaluation commands, research scripts and review tools. Checked syntax for all Python source files. Historical training and evaluation scripts received static review only: they require separately prepared local inputs and were not executed, retrained or used to inspect historical test predictions.

The repository was verified PUBLIC at the start of this review. Keep it public. Source visibility and Streamlit viewer access are separate settings. No model parameters, data splits, stored evaluation metrics or model artifacts changed.

## Confirmed issues fixed

1. UTF-8 CSV files containing null bytes could silently lose text in the pandas parser. Such files now fail with an actionable error before parsing.
2. Single/batch inference could choose a label from NaN, infinite, wrong-shaped or invalid probability output. Shared validation now rejects these outputs before any category suggestion or export is produced. Class order and distinct labels must match the probability columns.
3. Non-text single inputs raised a low-level Unicode TypeError. They now receive a normal input error. Programmatic batch inputs retain non-text records as explicit input-error rows.
4. The batch interface could expose a traceback, or leave an older batch visible, after an inference ValueError. It now clears the old result and displays the error without publishing a replacement prediction.
5. `review_all` responses incorrectly described a validation-selected threshold. The returned explanation now correctly says every suggestion needs human review.
6. The model cache used the local-registry timestamp even in hosted mode. Its key now includes the selected profile metadata, including its version/checksum, so registry changes invalidate the correct model cache.
7. The README still called the repository private after its visibility changed. It now states public.

## Validation

- 24 application/helper tests passed, including a real DATA100 Streamlit AppTest using three synthetic inputs and a deliberately injected batch failure.
- 11 independent review-tool tests passed.
- All tracked Python sources plus the new tests passed syntax compilation.
- Separate real DATA100 synthetic single/batch smoke checks passed; cloud mode does not import torch and the ordinary local registry still defaults to FT01.
- The shipped DATA100 SHA256 remained `e9fb35f684d264855799dbfac4b398e989ddb7790bd50ab90473c30d2c313bc2`.
- Tracked-path and credential-pattern scans found no raw-data directory, private review journal, FT01 weights or recognized embedded secret pattern. The only shared joblib is the explicitly authorized DATA100 cloud artifact. Pattern scans are not a comprehensive security guarantee.

## Limits

Passing tests does not prove that all bugs are absent. This review does not independently certify model accuracy, Linux/Python compatibility across all versions, every malformed MIME input, dependency vulnerabilities, provider retention, concurrency capacity or continuous free-host uptime. Historical research scripts were not dynamically rerun, and no full FT01 evaluation was repeated. Existing reused-validation and score-calibration limitations remain.

Use synthetic examples in the hosted application. Real complaints and private emails should not be submitted to the public demo. The preserved local FT01 application and model artifacts were not replaced by this GitHub review.
