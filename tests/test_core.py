"""Check leakage prevention and invalid inference inputs with synthetic fixtures."""

import unittest

import pandas as pd

from src.data import clean_text, prepare_data
from src.predict import predict_text
from src.review import review_flags, select_review_rule


class CoreTests(unittest.TestCase):
    def test_earliest_label_retained_and_duplicates_cannot_cross_splits(self):
        rows = [
            ("1", "2024-11-01", "A", "I do NOT owe 123 dollars."),
            ("2", "2024-12-14", "B", "  I do NOT owe 123 dollars.  "),
            ("3", "2024-11-02", "B", "A different training complaint."),
            ("4", "2024-12-15", "A", "A validation complaint."),
            ("5", "2024-12-25", "B", "A reserved complaint."),
        ]
        frame = pd.DataFrame(rows, columns=["Complaint ID", "Date received", "Issue", "Consumer complaint narrative"])
        frame["Product"] = "Debt collection"
        frame["Company"] = "Synthetic test company"
        splits, audit = prepare_data(frame)
        self.assertEqual(audit["duplicate_narrative_rows_removed"], 1)
        self.assertEqual(audit["conflicting_narrative_groups_audited"], 1)
        self.assertEqual(splits["train"].iloc[0]["issue"], "A")
        self.assertNotIn("2", pd.concat(splits.values())["complaint_id"].tolist())
        self.assertEqual([len(splits[name]) for name in ["train", "validation", "test"]], [2, 1, 1])

    def test_cleaning_keeps_negation_and_numbers(self):
        self.assertEqual(clean_text("  I do NOT owe 123 dollars.\n"), "i do not owe 123 dollars.")

    def test_empty_and_oversized_input_rejected_before_model_call(self):
        for text in [" \n\t", "a" * 50001]:
            with self.subTest(length=len(text)):
                with self.assertRaises(ValueError):
                    predict_text(text, None)

    def test_rule_chooses_maximum_eligible_coverage(self):
        truth = ["A"] * 100 + ["B"] * 50
        predicted = ["A"] * 150
        scores = [.8] * 100 + [.2] * 50
        rule, _ = select_review_rule(truth, predicted, scores)
        self.assertEqual(rule["threshold"], .225)
        self.assertEqual(rule["validation_selection"]["accepted_count"], 100)

    def test_no_eligible_threshold_means_review_all(self):
        rule, _ = select_review_rule(["A"] * 100, ["B"] * 100, [.9] * 100)
        self.assertEqual(rule["mode"], "review_all")
        self.assertTrue(all(review_flags([.1, .99], rule)))

    def test_review_boundary_and_nonfinite_values(self):
        flags = review_flags([.4499, .45, float("nan")], {"mode": "threshold", "threshold": .45})
        self.assertEqual(flags.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
