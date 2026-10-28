# Model and evaluation card

Task: seven-class English CFPB debt-collection Issue classification from narrative text only. Consumer-selected labels can be ambiguous. No Issue, Sub-issue or company-response field is used as an input feature.

Active application: v1, word/bigram TF-IDF (40,000-feature cap) and balanced logistic regression, lbfgs, max_iter1000, tol .0001. Recorded convergence:59 iterations; optimization was not shown to be incomplete. The app has not been switched to research candidates.

DATA100: same fixed D01 configuration, expanded training set19,223. Macro-F1 .4409 and Accuracy53.08% on the repeatedly used1,428-record validation. Written-notification F1 declined versus D01. No production adequacy claim is made.

Scores are uncalibrated model outputs, not verified correctness probabilities. Score-gap .10 is an interface heuristic. The legacy review threshold .45 does not imply auto-acceptance or human confirmation. High-score mistakes remain possible; low scores are not a reliable unknown-class detector.

Do not use the model for legal conclusions, autonomous complaint disposition or company quality rankings. It provides a starting suggestion that must be checked against the narrative. Saving reviewer decisions is not implemented.
