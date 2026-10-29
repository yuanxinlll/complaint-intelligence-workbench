"""Top-2 reporting from existing scores only, no fitting or inference."""
from pathlib import Path
import json,hashlib
import pandas as pd,numpy as np
O=Path(__file__).resolve().parent;R=O.parents[2];assert not (O/'top2_summary.csv').exists()
labels=json.loads((R/'research/discrimination_20261028/stage_4_experiments/DATA100/score_metadata.json').read_text())['labels_in_class_score_order'];sources={
'v1':R/'research/discrimination_20261028/call_sensitivity/v1_original/validation_predictions.csv',
'DATA100':R/'research/discrimination_20261028/stage_4_experiments/DATA100/validation_predictions.csv',
'AUG01':R/'research/discrimination_20261028/reviewed_augmentation/validation_predictions.csv'}
for epoch in [1,2,3]:
 p=R/f'research/discrimination_20261028/finetune_run/epoch_{epoch}/validation_predictions.csv'
 if p.exists():sources[f'DistilBERT_epoch{epoch}']=p
meta=json.loads((R/'research/discrimination_20261028/finetune_run/best_model/config.json').read_text());assert [meta['id2label'][str(i)] for i in range(7)]==labels
truth=pd.read_csv(R/'data/processed/validation.csv',usecols=['complaint_id','issue'],dtype=str).set_index('complaint_id');rows=[];byclass=[];manifest=[]
for name,path in sources.items():
 df=pd.read_csv(path,dtype={'complaint_id':str}).set_index('complaint_id');assert df.index.is_unique and len(df)==1428 and set(df.index)==set(truth.index);df=df.loc[truth.index];assert df.issue.equals(truth.issue)
 scores=df[[f'class_score_{j}' for j in range(7)]].to_numpy();assert np.isfinite(scores).all() and np.allclose(scores.sum(axis=1),1,atol=1e-6);order=np.argsort(-scores,axis=1,kind='stable');first=np.array(labels)[order[:,0]];second=np.array(labels)[order[:,1]];assert np.array_equal(first,df.predicted_issue)
 y=df.issue.to_numpy();one=first==y;two=one|(second==y);gap=scores[np.arange(len(df)),order[:,0]]-scores[np.arange(len(df)),order[:,1]];hint=gap+1e-12<.10;display=one|(hint&(second==y))
 def stats(mask):
  return {'n':int(mask.sum()),'top1_matches':int(one[mask].sum()),'top2_matches':int(two[mask].sum()),'second_only_matches':int((two[mask]&~one[mask]).sum()),'neither':int((~two[mask]).sum()),'top1_rate':float(one[mask].mean()),'top2_rate':float(two[mask].mean()),'ambiguity_hint_rows':int(hint[mask].sum()),'display_policy_candidate_matches':int(display[mask].sum()),'display_policy_candidate_rate':float(display[mask].mean())}
 rows.append({'model':name,**stats(np.ones(len(df),dtype=bool))})
 for label in labels:byclass.append({'model':name,'issue':label,**stats(y==label)})
 manifest.append({'model':name,'source':str(path.relative_to(R)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
pd.DataFrame(rows).to_csv(O/'top2_summary.csv',index=False);pd.DataFrame(byclass).to_csv(O/'top2_per_class.csv',index=False);(O/'top2_sources.json').write_text(json.dumps({'sources':manifest,'class_order':labels,'new_inference':0,'new_training':0,'no_model_selection_or_threshold_tuning':True},indent=2));print(pd.DataFrame(rows).to_string(index=False))
