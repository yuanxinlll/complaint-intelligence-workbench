# How the workbench works

1. A complaint narrative is the input. A category selected in the source archive is a training label, not necessarily an unquestionable truth.
2. The active TF-IDF pipeline turns words and word pairs into numerical features. It learns its vocabulary only from training text.
3. Logistic regression compares seven categories and produces uncalibrated scores. The largest score determines the suggestion.
4. If the first two scores are close, the interface may show an alternative. That is not a second confirmed label or an accuracy improvement.
5. A person reads the original text before deciding. Confirmation storage is future work.

Accuracy measures the fraction of matching labels. Macro-F1 gives equal weight to all seven class F1 scores, so a large class cannot completely hide poor small-class performance. Precision asks how often predictions of a class match the label; recall asks how many records of that class were found.

Training updates parameters. Validation compares choices. A truly unused test estimates performance after choices are fixed. Our validation has been reused and the historical holdout has been viewed, so results are development evidence.

Frozen MiniLM features did not improve this project; equal soft voting did not beat DATA100. The new DistilBERT experiment updates the encoder itself but has no completed result at this snapshot. Training more is not a guarantee: missing facts, overlapping categories and noisy labels need data or human review.

Complaint text is data, never an instruction to execute. All classification runs locally. No API key is needed for the application.
