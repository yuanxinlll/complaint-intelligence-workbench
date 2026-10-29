import unittest,tempfile,csv,json
from pathlib import Path
from types import SimpleNamespace
from blind_review_stats import read,stats
class Tests(unittest.TestCase):
 def setUp(self):self.t=tempfile.TemporaryDirectory();self.d=Path(self.t.name)
 def tearDown(self):self.t.cleanup()
 def sheet(self,name,rows,headers=None):
  p=self.d/name
  with p.open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=headers or ['sample_id','label','sufficiency']);w.writeheader();w.writerows(rows)
  return str(p)
 def runstats(self,a,b=None):return stats(SimpleNamespace(a=a,b=b,duplicates='error',jsonl_revisions='error',model=None,gold=None))
 def test_empty(self):
  a=self.sheet('a.csv',[]);b=self.sheet('b.csv',[]);r=self.runstats(a,b)['human_human'];self.assertIsNone(r['kappa']);self.assertEqual(r['message'],'No samples labelled by both reviewers')
 def test_duplicates_error_and_first(self):
  p=self.sheet('a.csv',[{'sample_id':'x','label':'A'},{'sample_id':'x','label':'B'}])
  with self.assertRaisesRegex(ValueError,'x'):read(p)
  self.assertEqual(read(p,duplicates='first')[0]['x']['labels'],{'A'})
 def test_multilabel_kappa_exclusion(self):
  a=self.sheet('a.csv',[{'sample_id':'1','label':'A|B'},{'sample_id':'2','label':'A'},{'sample_id':'3','label':'B'}]);b=self.sheet('b.csv',[{'sample_id':'1','label':'B|C'},{'sample_id':'2','label':'A'},{'sample_id':'3','label':'B'}]);r=self.runstats(a,b)['human_human'];self.assertEqual(r['any_overlap_count'],3);self.assertEqual(r['exact_set_agreement_count'],2);self.assertEqual(r['kappa_multilabel_excluded'],1);self.assertEqual(r['kappa'],1)
 def test_degenerate_kappa(self):
  a=self.sheet('a.csv',[{'sample_id':'1','label':'A'}]);b=self.sheet('b.csv',[{'sample_id':'1','label':'A'}]);r=self.runstats(a,b)['human_human'];self.assertIsNone(r['kappa']);self.assertNotIn('NaN',json.dumps(r))
 def test_existing_format_single(self):
  a=self.sheet('a.csv',[{'review_id':'R01','primary_category':'A','other_reasonable_categories':'B','multiple_reasonable':'False','insufficient_information':'True'}],['review_id','primary_category','other_reasonable_categories','multiple_reasonable','insufficient_information']);r=self.runstats(a);self.assertEqual(r['reviewer_A']['multiple_reasonable_count'],1);self.assertEqual(r['reviewer_A']['insufficient_count'],1);self.assertFalse(r['human_human']['computed'])
 def test_revision_chain(self):
  p=self.d/'a.jsonl';p.write_text('\n'.join(json.dumps(x) for x in [{'review_id':'R1','primary_category':'A','event_id':'e1'},{'review_id':'R1','primary_category':'B','event_id':'e2','supersedes_event_id':'e1'}]))
  with self.assertRaises(ValueError):read(p)
  self.assertEqual(read(p,revisions='latest')[0]['R1']['labels'],{'B'})
 def test_same_file_not_independent(self):
  a=self.sheet('a.csv',[{'sample_id':'1','label':'A'}])
  with self.assertRaises(ValueError):self.runstats(a,a)
 def test_missing_id(self):
  a=self.sheet('a.csv',[{'sample_id':'','label':'A'}])
  with self.assertRaises(ValueError):read(a)
 def test_author_not_reviewer(self):
  a=self.sheet('a.csv',[{'sample_id':'1','expected_author_only':'A'}],['sample_id','expected_author_only'])
  with self.assertRaises(ValueError):read(a)
 def check_existing_output(self,command):
  import subprocess,sys
  out=self.d/'existing.json';out.write_text('KEEP ORIGINAL')
  result=subprocess.run([sys.executable,str(Path(__file__).with_name('blind_review_stats.py')),*command,'--out',str(out)],text=True,capture_output=True)
  self.assertEqual(result.returncode,2);self.assertIn('output file already exists',result.stderr);self.assertIn('choose a new name',result.stderr);self.assertNotIn('Traceback',result.stderr);self.assertEqual(result.stdout,'');self.assertEqual(out.read_text(),'KEEP ORIGINAL')
 def test_existing_stats_output_friendly(self):self.check_existing_output(['stats','--a',str(self.d/'missing_input.csv')])
 def test_existing_sheet_output_friendly(self):self.check_existing_output(['make-sheet','--samples',str(self.d/'missing_input.csv')])
if __name__=='__main__':unittest.main()
