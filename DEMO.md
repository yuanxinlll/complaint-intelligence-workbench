# Complaint Intelligence Workbench

**Data + AI · Local complaint classification with human judgment**

A working prototype that turns English debt collection narratives into category suggestions while keeping the original text available for review.

### The problem

Complaint narratives are messy: long letters, repeated templates, missing context, and several concerns in one message. A single category can hide that uncertainty. We demonstrate an enterprise-style triage workflow with public CFPB historical data, rather than claim access to a company's private complaints.

### What works today

- Paste one narrative or import a small UTF-8 CSV.
- Receive one suggested category from seven debt collection issues.
- When the top-two class scores are less than 0.10 apart, see **Possible ambiguity** and an **Alternative category**. This is an unvalidated display heuristic, not a probability of error or a definitive ambiguity judgment.
- Read the original narrative beside the suggestion, inspect batch distributions, and export unconfirmed suggestions.
- Run classification locally without sending complaint text to an external AI API.

Every suggestion requires human confirmation. This prototype does not yet capture or persist human decisions. A larger score does not authorize automatic acceptance.

### How it works

Historical CSV → product and text checks → normalized exact-text deduplication → chronological development splits → training-only TF-IDF → balanced logistic regression → local Streamlit interface.

Only narrative text is a feature. Issue labels, company responses, and sub-issues are not prediction inputs. Near-duplicate templates and conflicting consumer labels remain limitations.

### Evidence, including what did not improve

**Active application: v1**, trained on 6,351 records. On the repeatedly used 1,428-record development validation set: **52.59% Accuracy; 0.3975 Macro-F1**. These are development results, not independent enterprise-performance evidence. Historical holdout data have also been viewed during development.

A separate word-only experiment, D03, reached 0.4017 Macro-F1 on the same validation set using a different 6,333-record training version. Its Top-2 inclusion rate was 69.47%; that is not Top-1 accuracy and is not a measurement of this active app. A frozen MiniLM experiment reached 0.3834 Macro-F1 and was not adopted. We kept negative results and did not replace the app to claim progress.

### Why this fits Data + AI

The contribution is a traceable path from messy data to a working review aid: explicit data boundaries, preserved model versions, honest evaluation, and visible uncertainty. Potential triage benefits have not yet been measured with operational users.

### Live demo

1. Paste a synthetic complaint and inspect the suggested category and original text.
2. If the ambiguity hint appears, compare the alternative; otherwise explain that the heuristic did not fire and errors are still possible.
3. Run the three synthetic batch examples and export suggestions that are explicitly unconfirmed.

### Next, only with approval

Human confirmation records, clearer label guidance, and separately authorized earlier-data experiments. No automatic retraining, mailbox connection, legal advice, or deployment is included.

### Latest offline research snapshot

Fixed-model historical-data expansion produced DATA100: Macro-F1 .4409, Accuracy53.08% on the reused validation set. It remains an offline candidate and does not power this demo. Equal soft voting did not improve it. A single DistilBERT task-fine-tuning run is ongoing, with no completed result claimed.
