# Research summary and limitations

## Data lineage

Original November–December2024 archive:591,058 rows inspected;11,706 debt-collection narrative candidates;2,529 excess normalized duplicates;9,177 retained records. Original training6,351, validation1,428, historical holdout1,398. Eighteen training-only records were quarantined with original labels retained, leaving6,333. Small-class training counts after quarantine were86 and140, not88 and140.

An official August–October2024 archive had824,514 rows and18,422 debt-collection narratives. Excluding3,679 overlaps with existing normalized text,93 rows in29 conflicting-label groups, and1,760 further internal duplicates produced12,890 additions. Half additions6,442 were selected by deterministic within-class SHA256 rank(seed42), nested in the full additions. Dates respect the original training boundary; validation was unchanged.

Exact normalized text and IDs were checked against original splits without using historical-test labels for selection. Near-template screening used five-word shingles,128-component MinHash,32 bands, bounded candidates and exact Jaccard>=.80 flags. Sixty validation records had a flagged similar training record. Flags did not delete records; similarity does not establish the same event. Candidate limits can miss matches. Human template adjudication has not been completed.

## Fixed experiments

| Experiment | Macro-F1 | Accuracy | Interpretation |
|---|---:|---:|---|
| D01 balanced word/bigram LR | .3931 |51.96%| Fixed baseline |
| D02 no class balancing | .3265 |56.16%| Higher accuracy concealed zero F1 for both small classes |
| D03 unigram LR | .4017 |50.28%| Historical validation-selected word-feature best |
| D04 character features | .3857 |49.09%| No improvement |
| D05 LinearSVC | .3846 |54.62%| No improvement in main metric |
| SEM01 frozen MiniLM + LR | .3834 |44.26%| This frozen representation did not improve performance |
| DATA50 | .4217 |53.15%| Fixed D01 +6,442 historical additions |
| DATA100 | .4409 |53.08%| Fixed D01 +12,890 additions |
| ENS01 .5 DATA100 + .5 SEM01 | .4384 |51.47%| Not recommended |

DATA50/100 converged without warnings in77/118 iterations; fits took7.493/14.107 seconds. DATA100 improved six class F1 scores versus D01 but slightly reduced written-notification F1. Its third-party-threat recall was5/25 and precision5/30, still poor. Three bidirectional confusion totals versus D01: debt-not-owed/false-statements151→153, debt-not-owed/legal-action112→81, debt-not-owed/written-notice100→101. Stage-B original-v1 totals151/108/98 refer to a different model and are not interchangeable.

DATA100/SEM01:523 both correct,235 word-only correct,109 semantic-only correct,341 both wrong with the same prediction,220 both wrong with different predictions. Oracle Accuracy60.71% is a theoretical answer-aware upper bound, not a deployed ensemble result. ENS01 corrected68 DATA100 errors but lost91 correct records.

Complaint-level paired bootstrap for DATA100 minus D01 Macro-F1 gave delta .0478, percentile95% interval[.0129,.0846]; repeated validation selection and template dependence can make this optimistic. Do not interpret it as independent final significance evidence.

## Completed task fine-tuning and integration

FT01 completed all three epochs. Accuracy 47.55%, 48.46%, 52.24%; Macro-F1 .4330, .4239, .4629. Epoch3 was selected by the predeclared Macro-F1 rule and is now the local app default. See README and evidence/ for complete per-class results. DATA100 and v1 remain selectable rollback versions.

FT01 improves electronic-communications F1 .3363→.4286 and third-party disclosure-threat F1 .1818→.3333, while all five other class F1 scores decrease. Overall Accuracy 52.24% is below DATA100 53.08%. Saved-score Top-2 is69.82%, below DATA10073.39%; this is candidate inclusion, not Top-1 improvement or reviewer success. Runtime and model size are substantially higher in the recorded implementation.

One fixed CPU run: AdamW2e-5, weight decay .01, seed42, float32, effective batch16, training-only inverse-frequency weights. Inputs over510 content tokens retain first255+last255 plus CLS/SEP; omitted middle content is disclosed. This differs from frozen MiniLM full-chunk pooling, so comparison does not isolate only encoder updating.

The terminal preservation assertion flagged only two Python cache files after the complete checkpoint/evaluations were saved. An independent audit verified22922 protected files, source/data/original weights unchanged, saved predictions/metrics consistent. Original failure logs remain local. Integration reloaded the checkpoint and verified three saved validation outputs (no model selection), plus synthetic input/CSV/email/invalid/long-text/export/rollback checks. FT01 keyword robustness has not been retested. No training was added during integration.

## What evidence cannot establish

Validation has been reused; the historical holdout has already been viewed. No new future-time independent final test has been acquired. More data also changed historical expression distribution, so its effect is not purely sample count. Error examples show missing context, multiple requests and category overlap, but their frequencies are not population estimates. No labels were automatically corrected. No real enterprise generalization, time savings or safe autonomous operation has been demonstrated.

## Full seven-class metrics for the best candidate and the unsuccessful ensemble

|Model|Issue|Precision|Recall|F1|Support|
|---|---|---:|---:|---:|---:|
|DATA100|Attempts to collect debt not owed|0.7339|0.5559|0.6326|635|
|DATA100|Communication tactics|0.4857|0.5667|0.5231|120|
|DATA100|Electronic communications|0.2676|0.4524|0.3363|42|
|DATA100|False statements or representation|0.4931|0.5296|0.5107|270|
|DATA100|Threatened to contact someone or share information improperly|0.1667|0.2000|0.1818|25|
|DATA100|Took or threatened to take negative or legal action|0.4076|0.4717|0.4373|159|
|DATA100|Written notification about debt|0.4095|0.5367|0.4645|177|
|ENS01|Attempts to collect debt not owed|0.7374|0.5087|0.6021|635|
|ENS01|Communication tactics|0.4710|0.5417|0.5039|120|
|ENS01|Electronic communications|0.2262|0.4524|0.3016|42|
|ENS01|False statements or representation|0.4931|0.5296|0.5107|270|
|ENS01|Threatened to contact someone or share information improperly|0.1667|0.3600|0.2278|25|
|ENS01|Took or threatened to take negative or legal action|0.3909|0.4843|0.4326|159|
|ENS01|Written notification about debt|0.4361|0.5593|0.4901|177|

## Reviewed synthetic augmentation and tooling

One reviewer completed14 synthetic cases; no second independent reviewer or human-ceiling estimate. Nine cases without unresolved author/reviewer disagreement were retained as base/neutral-phone pairs (18 additions); five ambiguous/unresolved cases were quarantined, preserving original answers. AUG01 reached .4433/53.29%, only+.0024 Macro-F1, below the recommendation gate. Those training-related probes are not independent tests. Review-statistics tooling passes11 synthetic tests and reports multilabel overlap separately from exact-set agreement; degenerate kappa remains undefined, not a fake zero/NaN.

Local EML import, input review, concise output and model-version-aware export are implemented. Final human-decision persistence, mailbox connection, automatic learning and deployment are not implemented.
