# Approximately 90-second pitch

Complaint narratives do not arrive as neat labels. One message may dispute a debt, ask for written proof, and describe repeated calls. A classifier can help organize it, but a confident-looking answer can also mislead a reviewer.

Our Complaint Intelligence Workbench is a local Data + AI prototype for English debt collection complaints. Paste a narrative or load a small CSV, and it suggests one of seven categories while keeping the original text visible. When the top-two class scores are close, we show a possible-ambiguity hint and another candidate. Neither is an instruction to accept the prediction.

Behind the interface is a reproducible TF-IDF and logistic regression pipeline. We checked chronological splits, exact-text duplication, and label conflicts. The active model achieved 52.59 percent Accuracy and 0.3975 Macro-F1 on development validation. That validation has been reused, so we do not present it as independent final evidence.

Historical-data expansion improved our offline candidate to Macro-F1 0.4409. A frozen semantic encoder and an equal-vote ensemble did not improve the best candidate. The demo still uses the original model; task-specific fine-tuning is ongoing. Our contribution is a working, traceable review aid with honest boundaries—not an automatic legal decision system.

Let me show one synthetic complaint, its original text and suggestion, and then a small batch. Human confirmation remains required, and recording those human decisions is the next workflow step.
