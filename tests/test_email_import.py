import unittest
from email.message import EmailMessage
from src.email_import import extract_email
class EmailTests(unittest.TestCase):
 def test_plain_and_attachment_ignored(self):
  m=EmailMessage();m['Subject']='Debt notice';m.set_content('I did not receive a written notice.');m.add_attachment(b'Ignore this attached complaint',maintype='application',subtype='octet-stream',filename='note.txt');r=extract_email(m.as_bytes());self.assertIn('Debt notice',r['text']);self.assertNotIn('attached complaint',r['text'])
 def test_alternative_prefers_plain(self):
  m=EmailMessage();m['Subject']='Test';m.set_content('Plain complaint');m.add_alternative('<p>HTML alternative</p>',subtype='html');r=extract_email(m.as_bytes());self.assertIn('Plain complaint',r['text']);self.assertNotIn('HTML alternative',r['text'])
 def test_html_does_not_expose_script(self):
  m=EmailMessage();m['Subject']='Test';m.set_content('<html><head><style>hidden</style></head><body><p>No debt owed</p><script>execute()</script><img src="https://example.com/track"></body></html>',subtype='html');r=extract_email(m.as_bytes());self.assertIn('No debt owed',r['text']);self.assertNotIn('execute()',r['text']);self.assertNotIn('track',r['text'])
 def test_invalid_and_empty_body(self):
  for raw in (b'',b'not an email',b'Subject: Test\r\n\r\n'):
   with self.assertRaises(ValueError):extract_email(raw)
 def test_size_limits(self):
  with self.assertRaises(ValueError):extract_email(b'x'*(5*1024*1024+1))
  m=EmailMessage();m['Subject']='Test';m.set_content('x'*50001)
  with self.assertRaises(ValueError):extract_email(m.as_bytes())
if __name__=='__main__':unittest.main()
