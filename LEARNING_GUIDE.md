# How it works

The workbench organizes English debt collection complaints into seven issue categories for human review. It runs locally using pandas, scikit-learn, and Streamlit.

## From text to prediction

1. **Normalize:** standardize Unicode, whitespace, and case while retaining numbers and negation.
2. **Vectorize:** TF-IDF assigns weights to words and two-word phrases, using up to 40,000 features learned from training data only.
3. **Classify:** logistic regression scores each category; the highest-scoring category becomes the prediction.
4. **Flag:** scores below the frozen threshold of 0.45 trigger a human review flag.

Training fits the vectorizer and classifier together in a scikit-learn pipeline:

```python
model.fit(train["narrative"], train["issue"])
```

The saved pipeline is reused for new complaints. Only narrative text enters the model; supplied labels, companies, and response outcomes are not input features. Clicking **Classify** never retrains the model.

## Data and evaluation

The CFPB November–December 2024 archive supplied 11,706 labeled debt collection narratives. Deduplication retained the earliest record for each normalized narrative, including its original label when later labels disagreed, leaving 9,177 records.

| Split | Dates (2024) | Records | Purpose |
|---|---|---:|---|
| Training | Nov 1–Dec 12 | 6,351 | Learn text features and category weights |
| Validation | Dec 13–21 | 1,428 | Select the review threshold |
| Test | Dec 22–31 | 1,398 | Evaluate the frozen model and rule |

Normalized duplicates cannot cross splits; similar templates may remain. The threshold maximizes validation coverage while requiring at least 100 unflagged records and 80% observed accuracy. If no threshold qualifies, all records are flagged. The rule is saved before test evaluation.

## Reading the output

- `predicted_issue`: the highest-scoring category.
- `model_score`: its uncalibrated model score, not a probability of correctness.
- `needs_review`: whether the review rule flagged the prediction.
- `original_issue`: an optional user-supplied label for comparison.
- `input_error`: why a batch row could not be classified.

Test accuracy was **48.50%**, compared with **46.57%** for always predicting the majority class. Macro-F1, which weights each category equally, improved from **0.0908** to **0.3888**.

Unflagged test records reached only **70.77%** accuracy, below the validation selection target. A false review flag means the score met the threshold, not that the prediction is verified. Labels are consumer-selected, and one complaint may contain multiple issues.

## Using the workbench

Enter a complaint or upload a CSV, classify it, then filter and export results. Invalid rows remain visible with an error. The distribution chart includes all valid batch rows, regardless of table filters. The three built-in examples are synthetic demonstrations, not evaluation evidence.

For implementation details, start with `src/predict.py`, then read `src/data.py`, `src/train.py`, and `src/evaluate.py`. The interface lives in `app.py` and `src/interface.py`.
