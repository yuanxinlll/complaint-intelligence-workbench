# Short English pitch

Complaint Intelligence Workbench helps a person review English debt-collection complaints. Paste text, review a locally imported email, or upload a small CSV. It suggests one of seven categories while keeping the original narrative available, and exports unconfirmed results.

The default task-fine-tuned DistilBERT model reached development Macro-F1 0.4629 and Accuracy 52.24%. Our fast word model reached0.4409 and53.08%. The semantic model improves two small classes but is not universally better. Both remain selectable; the validation has been reused, so we do not claim independent final performance.

We also measured concrete keyword sensitivity and documented failed fixes rather than hiding them. Human review remains necessary, scores are uncalibrated, and final confirmation storage is future work. Let me show one synthetic complaint and a small batch.
