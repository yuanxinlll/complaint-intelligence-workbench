# Blind-review statistics adapter

The user-supplied original scripts are preserved in `original/`. Only `blind_review_stats.py` was adapted; `topk_eval.py` was not rerun because saved-score Top-2 reporting already exists.

## Supported review formats

- Legacy CSV: `sample_id,label,sufficiency,confidence`, where sufficiency is `enough`, `insufficient` or `multiple`.
- Current app CSV: `review_id,primary_category,other_reasonable_categories,insufficient_information,multiple_reasonable,reason`. Extra columns are retained by the input file but not used as labels.
- Current app append-only JSONL with `event_id`, optional `supersedes_event_id`, `packet_sha256` and reviewer identity.

Labels are full category names. Generic multiple labels can be separated with `|`; existing app exports also support `;` and JSON arrays. Other reasonable categories are combined with the primary label. Exact-set agreement and any-overlap agreement are reported separately. A single prediction matching one allowed human label counts as overlap, not exact-set equality.

Duplicate sample IDs fail by default and list the IDs. `--duplicates first` explicitly keeps the first row; never silently overwrite. For real app revision journals use `--jsonl-revisions latest`: only linked revisions superseding the preceding event are accepted, with one reviewer and packet version. Unlinked repeats still fail. No existing review file is modified.

## Single reviewer

```bash
python blind_review_stats.py stats --a /path/to/review.csv --out /path/to/new_results.json
```

For the app journal, add `--jsonl-revisions latest`. Optional `--model` accepts `sample_id,model_label` or `review_id,predicted_issue` with matching diagnostic IDs; no model is loaded. Optional `--gold` accepts a genuine CFPB original `issue` reference with matching IDs, not synthetic author expectations. No reference is manufactured when samples are synthetic.

Single-reviewer outputs: counts/proportions marked insufficient or multiple among all unique samples, optional exact/overlap reference agreement, and explicit notice that human-human agreement and kappa require a second independent reviewer. These are not expert accuracy or ceiling estimates.

## Two independent reviewers

```bash
python blind_review_stats.py stats --a reviewer_a.csv --b reviewer_b.csv --out new_pair_results.json
```

One file per reviewer on the same packet. Same file or known same reviewer identity is rejected; unknown identity cannot establish independence, which must be ensured in collection. Original author labels cannot substitute for reviewer B. Empty labelled intersection returns “No samples labelled by both reviewers” and null metrics, never division-by-zero or NaN.

Kappa uses only pairs with a single label on both sides, excluding rows marked multiple. Excluded count is explicit. A degenerate expected agreement of one yields null plus an explanation. Exact-set and overlap agreement still include multi-label pairs. Insufficiency counts include blank-labelled cases, so uncertainty is not silently dropped. Comparisons use matched IDs and report missing IDs.

## Blank sheet and tests

```bash
python blind_review_stats.py make-sheet --samples samples.csv --out new_blank_sheet.csv
python -m unittest discover -s . -p 'test_*.py' -v
```

Blank-sheet inputs accept `sample_id,narrative` or `review_id,text`; output excludes author labels and predictions. Existing output paths are not overwritten. `SYNTHETIC_FORMAT_EXAMPLE.csv` is only a format example, not completed human evidence.

Actual local results are in `outputs/single_reviewer_stats.json`. Model comparison reuses saved DATA100 synthetic predictions with exact-text checks; no new inference. Fourteen synthetic cases and one reviewer cannot estimate overall complaint accuracy, inter-rater agreement or a model-performance ceiling. No new training, validation tuning, historical-test access, app change, GitHub push or upload.

## Output protection and portable bundle v2

Existing output names are checked before calculation or reading input data. Both stats and make-sheet exit with status2 and the friendly message “output file already exists, not overwritten ... (choose a new name)”. A concurrent file-creation race is also caught; exclusive writing still prevents overwriting. Results are printed only after an output file was successfully created. Eleven synthetic tests pass, including both existing-output cases.

The portable zip contains only the adapted script, tests, this README and a synthetic format example. The `original/` directory and `outputs/single_reviewer_stats.json` mentioned above refer to the local project archive; they are intentionally NOT inside the portable bundle. Actual reviewer identities and answers are excluded.
