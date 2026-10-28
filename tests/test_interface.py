"""Stage 4 checks for CSV boundaries and retained invalid input rows."""

import unittest

import numpy as np

from src.interface import classify_batch, read_upload


class FakeModel:
    classes_ = np.array(["A", "B"])

    def predict_proba(self, texts):
        return np.array([[.4, .6] for _ in texts])


class InterfaceTests(unittest.TestCase):
    def test_official_fields_and_original_labels(self):
        frame = read_upload(b'Consumer complaint narrative,Issue\nExample text,Original label\n')
        self.assertEqual(frame.columns.tolist(), ["narrative", "original_issue"])
        result = classify_batch(frame, FakeModel(), {"mode": "threshold", "threshold": .7})
        self.assertEqual(result.loc[0, "original_issue"], "Original label")
        self.assertEqual(result.loc[0, "predicted_issue"], "B")
        self.assertTrue(result.loc[0, "needs_review"])

    def test_invalid_rows_retained_without_prediction(self):
        frame = read_upload(b'complaint_id,narrative\n1,valid text\n2,   \n')
        result = classify_batch(frame, FakeModel(), {"mode": "threshold", "threshold": .45})
        self.assertEqual(len(result), 2)
        self.assertEqual(result.loc[1, "predicted_issue"], "")
        self.assertEqual(result.loc[1, "input_error"], "Empty narrative")
        self.assertTrue(result.loc[1, "needs_review"])

    def test_bad_encoding_missing_column_and_duplicate_headers(self):
        for content in [b'\xff\xfe', b'wrong\nvalue\n', b'narrative,narrative\na,b\n', b'narrative,predicted_issue\na,b\n']:
            with self.subTest(content=content):
                with self.assertRaises(ValueError):
                    read_upload(content)

    def test_row_limit_and_export_roundtrip(self):
        with self.assertRaises(ValueError):
            read_upload(('narrative\n' + 'text\n' * 501).encode())
        frame = read_upload('narrative,issue\n"A comma, and 中文",A\n'.encode('utf-8-sig'))
        result = classify_batch(frame, FakeModel(), {"mode": "review_all"})
        encoded = result.to_csv(index=False).encode('utf-8-sig')
        self.assertTrue(encoded.startswith(b'\xef\xbb\xbf'))
        self.assertIn('A comma, and 中文', encoded.decode('utf-8-sig'))


if __name__ == "__main__":
    unittest.main()
