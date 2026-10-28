# Shared aggregate results

The current research summary is [here](../docs/RESEARCH_SUMMARY.md). App v1 validation Accuracy52.59%, Macro-F1 .3975; best offline DATA10053.08%/.4409. These describe different training versions and must not be substituted for each other.

Historical v1 holdout:1,398 records, Accuracy48.50%, Macro-F1 .3888; most-frequent baseline46.57%/.0908. This holdout was viewed during development and is no longer a fresh independent test. The legacy rule left325 records unflagged,70.77% matching labels and95 errors; it does not justify auto-acceptance.

Existing figure files are historical artifacts, not current candidate evaluation. Do not interpret a review curve as improving the classifier itself. All suggestions still need human confirmation.
