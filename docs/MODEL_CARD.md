# Active model card

Default: ciw-distilbert-ft01-epoch3, English seven-class CFPB debt-collection Issue suggestions, trained on19,223 historical narratives. Task inputs: narrative text only. Development validation1428 records, Accuracy 52.24%, Macro-F1 .4629. The same validation has been repeatedly used, including three-epoch checkpoint selection. No fresh independent final test exists.

Two small-class F1 scores improve over DATA100 while five decrease. Overall Accuracy and Top-2 are lower than DATA100. Model is heavier and current CPU inference slower. Scores are uncalibrated, no auto-acceptance threshold. Every prediction needs human review.

Tokenizer/model are stored locally; max512 with first255+last255 content tokens when over510. Middle content omitted and flagged. Local EML concatenates reviewed subject and body, which may differ from complaint-only training inputs. No email-domain performance claim. Phone-keyword robustness of FT01 is not yet evaluated.

DATA100 and original v1 are selectable local rollback versions. Checksums and profile metrics are in config/model_profiles.json; complete class metrics in docs/evidence/FINAL_PER_CLASS.csv. No weights or data are committed. Not for autonomous legal decisions, complaint disposition or company quality rankings. Human confirmation records remain future work.
