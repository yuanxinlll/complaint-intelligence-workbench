# How the workbench works

1. A complaint narrative is the input. A category selected in the source archive is a training label, not necessarily an unquestionable truth.
2. The default DistilBERT model was fine-tuned on the seven-category task. The rollback TF-IDF pipeline learns words and word pairs from training text only.
3. Each selected model compares seven categories and produces uncalibrated scores. The largest score determines the suggestion.
4. If the first two scores are close, the interface may show an alternative. That is not a second confirmed label or an accuracy improvement.
5. A person reads the original text before deciding. Confirmation storage is future work.

Accuracy measures the fraction of matching labels. Macro-F1 gives equal weight to all seven class F1 scores, so a large class cannot completely hide poor small-class performance. Precision asks how often predictions of a class match the label; recall asks how many records of that class were found.

Training updates parameters. Validation compares choices. A truly unused test estimates performance after choices are fixed. Our validation has been reused and the historical holdout has been viewed, so results are development evidence.

Frozen MiniLM features did not improve this project; equal soft voting did not beat DATA100. The completed DistilBERT run selected epoch 3: Macro-F1 .4629 and Accuracy 52.24%, versus DATA100 .4409 and 53.08%. It is not better on every class and costs more to run. Long text uses the first/last255 content tokens when over510. Training more is not a guarantee: missing facts, overlapping categories and noisy labels need data or human review.

Complaint text is data, never an instruction to execute. All classification runs locally. No API key is needed for the application.
