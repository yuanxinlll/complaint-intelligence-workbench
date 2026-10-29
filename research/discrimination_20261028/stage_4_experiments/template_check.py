"""Bounded MinHash/LSH template screening. No fit, prediction or relabeling."""
from pathlib import Path
import os
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
import json,hashlib,time,re,resource,sys
from collections import defaultdict,Counter
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[3];O=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from src.data import clean_text

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):(O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def log(text):
 print(text,flush=True)
 with (O/'template.log').open('a') as f:f.write(text+'\n')
def limit():
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>6_000_000_000:raise RuntimeError('Template screen peak RSS >6GB')
 if time.monotonic()-started>1800:raise RuntimeError('Template screen >30min')

if __name__=='__main__':
 started=time.monotonic();assert not (O/'plan.json').exists(),'Preserve prior run'
 source=R/'research/discrimination_20261028/stage_4_data/unique_training_candidates.csv';inspection=json.loads((source.parent/'inspection.json').read_text());assert sha(source)==inspection['candidate_file_sha256']
 originals={}
 for base in ['src','data','artifacts','diagnostics']:
  for p in (R/base).rglob('*'):
   if p.is_file() and '__pycache__' not in p.parts:originals[str(p.relative_to(R))]=sha(p)
 for p in [R/'app.py',R/'PROJECT_STATUS.md',source]:originals[str(p.relative_to(R))]=sha(p)
 save('protected_before.json',originals)
 paths={'base':R/'diagnostics/stage_c_20261028/data/train.csv','validation':R/'data/processed/validation.csv','new':source}
 frames={name:pd.read_csv(path,dtype=str,keep_default_na=False) for name,path in paths.items()}
 assert [len(frames[x]) for x in ['base','validation','new']]==[6333,1428,12890]
 plan={'authorized':'template check plus exactly DATA50 and DATA100 fits, no app switch','seed':42,'data_sha256':{name:sha(path) for name,path in paths.items()},'template_screen':{'shingles':'five consecutive Unicode word tokens after clean_text, punctuation excluded only for similarity screening; original text unchanged','minhash_components':128,'hash':'32-bit blake2b shingle hashes; 128 fixed affine permutations modulo4294967311','bands':32,'rows_per_band':4,'bucket_over_500':'do not expand; report skipped bucket memberships','max_candidates_per_record':50,'candidate_order':'lexical source ID then source split','jaccard_threshold':.8,'jaccard_on':'exact string shingle sets, not the 32-bit hashed values','action':'flag only; no deletion or label change','review_sample':'10 at >=.90, 10 at .75 to <.90, 10 below .75 from screened candidates; ID/hash deterministic rank; not population sample','max_sample_pairs':30},'sampling':{'by_class':'floor(n/2) added at half, all at full','rank':"SHA256('seed42|' + source_id + '|' + normalized_text_hash)",'nested':True},'models':['clone saved D01 unfitted for DATA50','clone saved D01 unfitted for DATA100'],'fit_count':2,'test':'no historical-test inference or label access; validation remains1428','metric_rule':{'macro_f1_gain_over_D01_min':.01,'majority_f1_drop_max':.03,'each_minority_f1_drop_max':.02,'no_auto_replacement':True},'resources':{'CPU_threads':4,'fit_total_seconds_max':1800,'peak_RSS_bytes_max':6000000000,'new_processed_bytes_max':1000000000},'script_sha256':sha(__file__)}
 save('plan.json',plan)
 docs=[]
 for split,frame in frames.items():
  for row in frame.itertuples():docs.append({'split':split,'id':row.complaint_id,'issue':row.issue,'text':row.narrative})
 identities=[(d['split'],d['id']) for d in docs];assert len(set(identities))==len(docs)
 rng=np.random.default_rng(42);a=rng.integers(1,2**32,size=128,dtype=np.uint64);b=rng.integers(0,2**32,size=128,dtype=np.uint64);prime=np.uint64(4294967311)
 sets=[];signatures=np.full((len(docs),128),np.iinfo(np.uint64).max,dtype=np.uint64);short=0
 for i,d in enumerate(docs):
  tokens=re.findall(r'(?u)\b\w+\b',clean_text(d['text']));ss={' '.join(tokens[j:j+5]) for j in range(len(tokens)-4)};sets.append(ss)
  if ss:
   values=np.array([int.from_bytes(hashlib.blake2b(s.encode(),digest_size=4).digest(),'little') for s in ss],dtype=np.uint64)
   signatures[i]=((values[:,None]*a[None,:]+b[None,:])%prime).min(axis=0)
  else:short+=1
  if i%2500==0:log(f'MinHash {i}/{len(docs)}');limit()
 buckets=defaultdict(list)
 for i,ss in enumerate(sets):
  if not ss:continue
  for band in range(32):buckets[(band,signatures[i,band*4:(band+1)*4].tobytes())].append(i)
 saturated=sum(len(v)>500 for v in buckets.values());saturated_memberships=0;cap_records=0;pairs=set();raw_candidates=0
 for i,ss in enumerate(sets):
  if not ss:continue
  candidates=set()
  for band in range(32):
   bucket=buckets[(band,signatures[i,band*4:(band+1)*4].tobytes())]
   if len(bucket)>500:saturated_memberships+=1;continue
   candidates.update(bucket)
  candidates.discard(i);raw_candidates+=len(candidates)
  if len(candidates)>50:cap_records+=1
  for j in sorted(candidates,key=lambda j:(docs[j]['id'],docs[j]['split']))[:50]:pairs.add((min(i,j),max(i,j)))
  if i%5000==0:limit()
 del buckets
 log(f'Exact Jaccard checks: {len(pairs)} candidate pairs; capped records {cap_records}, saturated buckets {saturated}')
 findings=[]
 for n,(i,j) in enumerate(sorted(pairs)):
  left,right=sets[i],sets[j];inter=len(left&right);score=inter/(len(left)+len(right)-inter)
  if score>=.8 or n<0:pass
  findings.append({'left_index':i,'right_index':j,'left_id':docs[i]['id'],'right_id':docs[j]['id'],'left_split':docs[i]['split'],'right_split':docs[j]['split'],'left_issue':docs[i]['issue'],'right_issue':docs[j]['issue'],'jaccard':score,'flagged_ge_0_80':score>=.8})
  if n%20000==0:limit()
 table=pd.DataFrame(findings);table.to_csv(O/'template_candidate_pairs.csv',index=False,mode='x')
 flagged=table.loc[table.flagged_ge_0_80];flagged_ids=set();crossval_ids=set();groups=Counter()
 for row in flagged.itertuples():
  flagged_ids.update([row.left_index,row.right_index]);groups['--'.join(sorted([row.left_split,row.right_split]))]+=1
  if row.left_split=='validation' and row.right_split!='validation':crossval_ids.add(row.left_id)
  if row.right_split=='validation' and row.left_split!='validation':crossval_ids.add(row.right_id)
 counts=[]
 for split,frame in frames.items():
  for label in sorted(frame.issue.unique()):
   total=sum(d['split']==split and d['issue']==label for d in docs);marked=sum(docs[i]['split']==split and docs[i]['issue']==label for i in flagged_ids)
   counts.append({'split':split,'issue':label,'rows':total,'flagged_template_candidate_rows':marked,'fraction':marked/total})
 pd.DataFrame(counts).to_csv(O/'template_flags_by_class.csv',index=False,mode='x')
 sample=[]
 for stratum,mask in [('high_ge_0_90',table.jaccard.ge(.9)),('borderline_0_75_to_lt_0_90',table.jaccard.ge(.75)&table.jaccard.lt(.9)),('below_0_75',table.jaccard.lt(.75))]:
  subset=table.loc[mask].copy();subset['rank']=[hashlib.sha256(f'seed42|{r.left_id}|{r.right_id}'.encode()).hexdigest() for r in subset.itertuples()]
  for row in subset.sort_values('rank').head(10).itertuples():
   info={k:getattr(row,k) for k in ['left_id','right_id','left_split','right_split','left_issue','right_issue','jaccard']}
   info.update(stratum=stratum,left_text=docs[row.left_index]['text'],right_text=docs[row.right_index]['text'],human_review_completed=False,reviewed_same_event=None,reviewed_shared_template=None,review_reason='')
   sample.append(info)
 save('template_review_pairs.json',sample)
 stats={'docs':len(docs),'candidate_pairs_checked':len(table),'flagged_pairs':len(flagged),'flagged_pair_types':dict(groups),'flagged_unique_rows':len(flagged_ids),'validation_rows_flagged_against_training':len(crossval_ids),'validation_flagged_ids':sorted(crossval_ids),'short_under_five_words':short,'saturated_buckets_skipped':saturated,'saturated_query_memberships':saturated_memberships,'records_candidate_cap_reached':cap_records,'raw_query_candidate_memberships':raw_candidates,'cap_semantics':'max50 candidates per query; a row can still be reached from another query','screened_pair_file_sha256':sha(O/'template_candidate_pairs.csv'),'sample_pairs':len(sample),'human_review_completed':0,'interpretation':'Possible shared templates; not identical complaints, not expert labels or reliable population template rate. Candidate caps/LSH can miss pairs.','data_changed':False,'elapsed_seconds':time.monotonic()-started,'peak_RSS_bytes_macOS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
 save('template_summary.json',stats);log(json.dumps({k:v for k,v in stats.items() if k!='validation_flagged_ids'},ensure_ascii=False))
