# Evaluation results

Recorded evaluation: October 28, 2026. Data checksums were verified against the training audit. The review rule was selected on validation data and saved before evaluating the reserved test set; test results did not change the model or threshold.

## Reserved test performance

Test period: December 22–31, 2024; 1,398 records.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Majority-class baseline | 46.57% | 0.0908 |
| TF-IDF + logistic regression | 48.50% | 0.3888 |

Accuracy measures agreement with consumer-selected labels. Macro-F1 averages class-level F1 scores equally. The text model recognizes more categories than the baseline, but overall accuracy remains limited.

| Issue | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Attempts to collect debt not owed | 0.671 | 0.564 | 0.613 | 651 |
| Communication tactics | 0.404 | 0.579 | 0.476 | 95 |
| Electronic communications | 0.321 | 0.191 | 0.240 | 47 |
| False statements or representation | 0.372 | 0.394 | 0.383 | 226 |
| Threatened to contact someone or share information improperly | 0.444 | 0.160 | 0.235 | 25 |
| Took or threatened to take negative or legal action | 0.311 | 0.457 | 0.370 | 151 |
| Written notification about debt | 0.392 | 0.419 | 0.405 | 203 |

![Test confusion matrix](test_confusion_matrix.png)

Rows are consumer-selected labels; columns are predictions. Smaller classes have limited support and weaker performance. Labels are not expert-verified ground truth.

## Human review rule

Thresholds from 0.15 to 0.975 were tested in increments of 0.025. Selection required at least 100 unflagged validation records and 80% observed accuracy, then maximized coverage. If no threshold qualified, all records would be flagged.

The frozen threshold is **0.45**: lower scores trigger review; equal scores do not.

| Metric | Validation | Reserved test |
|---|---:|---:|
| Records | 1,428 | 1,398 |
| Unflagged records | 368 | 325 |
| Unflagged coverage | 25.77% | 23.25% |
| Unflagged accuracy | 81.52% | 70.77% |
| Unflagged Macro-F1 (all 7 classes) | 0.5860 | 0.6278 |
| Flagged for review | 1,060 | 1,073 |
| Share of all errors flagged | 89.96% | 86.81% |
| Unflagged errors | 68 | 95 |

**Test performance missed the 80% target.** Scores are uncalibrated and do not reliably identify unknown categories. Coverage describes unflagged predictions, not actual automated handling. Validation threshold selection can produce optimistic estimates; subset metrics also reflect a different class mix from the full test set.

![Validation review curve](validation_review_curve.png)

## Error analysis

The five highest-scoring test errors are saved locally to expose confident mistakes rather than showcase successes. Examples labeled “Attempts to collect debt not owed” were predicted as negative/legal action (0.8795), written notification (0.8785), and false statements (0.7699), all unflagged.

Some narratives contain multiple concerns, and the two highest-scoring errors use similar templates. Exact normalized deduplication cannot remove all near-duplicates. These observations do not establish error causes.

## Local artifacts

- `artifacts/test_metrics.json`: class metrics, confusion matrix, and review performance.
- `artifacts/test_predictions.csv`: narratives, labels, predictions, scores, and flags.
- `artifacts/error_examples.json`: selected high-score errors.
- `artifacts/review_rule.json`: frozen rule and matching model SHA-256.
- `artifacts/evaluation.log`: progress without complaint text.

The repository shares summaries and figures, excluding models and row-level complaint data. Results are limited by label noise, class imbalance, public-sample selection bias, similar templates, and the short date range.
