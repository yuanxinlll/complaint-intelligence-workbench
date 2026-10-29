# Complaint Intelligence Workbench

**Local category suggestions with human review.**

Paste an English debt-collection complaint, review the text extracted from a local EML email, or import a small CSV. The workbench suggests one of seven categories, may show an alternative when scores are close, summarizes the batch and exports unconfirmed suggestions. A person must inspect the original text; final human decisions are not stored yet.

### What powers the demo

Default FT01 is a task-fine-tuned DistilBERT checkpoint from epoch 3. Development Macro-F1 is **0.4629**, Accuracy **52.24%** on 1,428 repeatedly used validation records. The selector retains DATA100 (**0.4409**, **53.08%**) and original v1. FT01 improves two small-class F1 scores but lowers five others and costs more to run; it is not universally better.

Long FT01 inputs use the first and last 255 content tokens when over 510; middle content is omitted and flagged. Scores are uncalibrated. Every suggestion requires human confirmation, regardless of score or ambiguity hint.

### Research evidence and failures

Historical-data expansion improved the word baseline. Frozen MiniLM and equal soft voting did not beat DATA100. Call/phone suppression and a fallback rule did not improve aggregate validation performance. A tiny reviewed-synthetic augmentation gain remained below its gate. Those failures were retained. FT01 phone robustness is not yet measured.

Validation has been reused, the historical holdout has been viewed, and there is no new independent final test. Fourteen synthetic cases reviewed by one person do not establish a human ceiling or kappa. Enterprise time savings and email-domain accuracy have not been measured.

### Local scope

No external model API, mailbox connection, cloud deployment or automatic online learning. Raw complaints and weights are not in the repository. Similar-history retrieval and stored human corrections remain proposed future work.
