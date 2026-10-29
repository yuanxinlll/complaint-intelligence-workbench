"""Exercise the shipped DATA100 app without network or real complaint data."""
import os
import unittest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from src.runtime_models import registry

class CloudAppTests(unittest.TestCase):
    def test_cloud_batch_error_is_friendly_and_clears_previous_results(self):
        with patch.dict(os.environ, {'CIW_PUBLIC_DEMO':'1'}):
            self.assertEqual(list(registry()['profiles']), ['DATA100'])
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1]/'cloud_demo/streamlit_app.py'), default_timeout=30).run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any('hosting server' in message.value for message in app.warning))
            next(b for b in app.button if b.label=='Try 3 synthetic examples').click().run()
            next(b for b in app.button if b.label=='Classify batch').click().run()
            self.assertEqual(len(app.exception), 0)
            result=app.session_state['batch_result']
            self.assertEqual(len(result), 3)
            self.assertTrue(result['model_version'].eq('ciw-lr-data100').all())
            self.assertTrue(result['confirmation_status'].eq('unconfirmed').all())
            with patch('src.interface.classify_batch', side_effect=ValueError('Synthetic invalid model scores')):
                next(b for b in app.button if b.label=='Classify batch').click().run()
            self.assertEqual(len(app.exception), 0)
            self.assertTrue(any('Synthetic invalid model scores' in message.value for message in app.error))
            self.assertNotIn('batch_result', app.session_state)

if __name__=='__main__':unittest.main()
