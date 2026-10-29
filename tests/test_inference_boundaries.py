"""Regression checks for malformed public inputs and invalid classifier output."""
import unittest
import numpy as np
import pandas as pd
from src.interface import read_upload, classify_batch
from src.predict import predict_text, candidate_details

class Model:
    classes_ = np.array(['A', 'B'])
    def __init__(self, scores): self.scores = scores
    def predict_proba(self, texts): return self.scores

class InferenceBoundaryTests(unittest.TestCase):
    def test_null_bytes_rejected_before_csv_parser_can_truncate_text(self):
        with self.assertRaisesRegex(ValueError, 'null bytes'):
            read_upload(b'narrative\nI do not\x00 owe this debt\n')

    def test_non_text_single_input_has_friendly_error(self):
        for value in [None, 123, float('nan')]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                predict_text(value, None)

    def test_bad_model_output_never_becomes_single_or_batch_suggestion(self):
        outputs = [np.array([[np.nan, .6]]), np.array([[np.inf, .6]]),
                   np.array([[.2, .3]]), np.array([[-.1, 1.1]]),
                   np.array([.4, .6]), np.array([[1.]]), np.array([[.4, .6], [.4, .6]])]
        for scores in outputs:
            model = Model(scores)
            with self.subTest(scores=scores.tolist()):
                with self.assertRaises(ValueError):
                    predict_text('Synthetic complaint', model, {'mode':'review_all'})
                with self.assertRaises(ValueError):
                    classify_batch(pd.DataFrame({'narrative':['Synthetic complaint']}), model, {'mode':'review_all'})

    def test_batch_keeps_non_text_rows_as_errors(self):
        result = classify_batch(pd.DataFrame({'narrative':[None, 12, 'Synthetic complaint']}), Model(np.array([[.4,.6]])), {'mode':'review_all'})
        self.assertEqual(result.input_error.tolist(), ['Narrative must be text', 'Narrative must be text', ''])
        self.assertEqual(result.predicted_issue.tolist(), ['', '', 'B'])

    def test_review_all_message_does_not_claim_validation_threshold(self):
        result = predict_text('Synthetic complaint', Model(np.array([[.4,.6]])), {'mode':'review_all'})
        self.assertIn('Every suggestion', result['review_status'])
        self.assertNotIn('validation-selected', result['review_status'])
        self.assertTrue(result['needs_review'])

    def test_candidates_reject_invalid_categories(self):
        with self.assertRaises(ValueError): candidate_details([.5,.5], ['A','A'])
        with self.assertRaises(ValueError): candidate_details([1.], ['A'])
        with self.assertRaises(ValueError): candidate_details(.5, ['A','B'])

if __name__=='__main__': unittest.main()
