from pathlib import Path
import sys,json,hashlib,importlib.util,time
import pandas as pd,numpy as np,joblib
from sklearn.base import clone
from sklearn.metrics import classification_report,accuracy_score,f1_score,confusion_matrix
from threadpoolctl import threadpool_limits
O=Path(__file__).resolve().parent;R=O.parents[2];sys.path.insert(0,str(R));assert not (O/'plan.json').exists(),'Do not repeat or overwrite'
from src.data import clean_text
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
reviewroot=R/'research/discrimination_20261028/human_review_14';source=reviewroot/'analysis/review_analysis.csv';df=pd.read_csv(source).fillna('');packet_sha=sha(reviewroot/'review_items.csv');journal=list((reviewroot/'reviews').glob('*.jsonl'));assert len(journal)==1
latest={}
for line in journal[0].read_text().splitlines():
 e=json.loads(line)
 if e['packet_sha256']==packet_sha:latest[e['review_id']]=e
assert len(latest)==14
mapping=pd.read_csv(reviewroot/'coordinator_only/author_expectations.csv').set_index('review_id');items=pd.read_csv(reviewroot/'review_items.csv').set_index('review_id');eligible=[];quarantine=[]
for rid,e in latest.items():
 reasons=[]
 if e.get('insufficient_information'):reasons.append('insufficient information')
 if e.get('multiple_reasonable') or e.get('other_reasonable_categories'):reasons.append('multiple reasonable labels')
 if e['primary_category']!=mapping.loc[rid,'expected_author_only']:reasons.append('reviewer/author boundary disagreement unresolved')
 if e['neutral_phone_changes_category']!='No':reasons.append('invariance not confirmed')
 if reasons:quarantine.append({'review_id':rid,'human_primary':e['primary_category'],'author_expectation':mapping.loc[rid,'expected_author_only'],'reason':'; '.join(reasons)})
 else:
  for version,col in [('base','text'),('one_neutral_phone','comparison_text')]:eligible.append({'complaint_id':f'SYNTH_REVIEW_{rid}_{version}','received_date':'','product':'Debt collection','issue':e['primary_category'],'narrative':items.loc[rid,col],'company':'','text_hash':hashlib.sha256(clean_text(items.loc[rid,col]).encode()).hexdigest(),'source_type':'human_reviewed_synthetic','source_review_id':rid,'variant':version})
assert len(eligible)==18 and len(quarantine)==5
basepath=R/'research/discrimination_20261028/stage_4_experiments/data/DATA100_train.csv';valpath=R/'data/processed/validation.csv';modelpath=R/'research/discrimination_20261028/stage_4_experiments/DATA100/model.joblib'
base=pd.read_csv(basepath,dtype=str,keep_default_na=False);val=pd.read_csv(valpath,dtype=str,keep_default_na=False);add=pd.DataFrame(eligible)
guard=set(base.text_hash)
for split in ['train','validation','test']:guard.update(pd.read_csv(R/f'data/processed/{split}.csv',usecols=['text_hash'],dtype=str).text_hash)
assert not set(add.text_hash)&guard;assert add.text_hash.is_unique;assert not set(add.complaint_id)&set(base.complaint_id)
add.to_csv(O/'synthetic_additions.csv',index=False);pd.DataFrame(quarantine).to_csv(O/'quarantine.csv',index=False)
train=pd.concat([base,add],ignore_index=True).fillna('');train.to_csv(O/'train.csv',index=False)
counts=pd.DataFrame({'base':base.issue.value_counts(),'synthetic_added':add.issue.value_counts()}).fillna(0).astype(int);counts['total']=counts.sum(axis=1);counts.to_csv(O/'class_counts.csv')
plan={'id':'AUG01','fit_count':1,'base_model_sha256':sha(modelpath),'base_train_sha256':sha(basepath),'new_train_sha256':sha(O/'train.csv'),'validation_sha256':sha(valpath),'review_journal_sha256':sha(journal[0]),'review_packet_sha256':packet_sha,'source_analysis_sha256':sha(source),'rules':'quarantine ambiguity or unresolved reviewer/author disagreement; both agreement does not establish expert truth; no automatic relabeling','additions':18,'synthetic_groups':9,'train_rows':len(train),'configuration':'clone fixed DATA100 unfitted; no coefficient attenuation, oversampling, sample weights or parameter search','synthetic_dates':'blank; authored2026 diagnostic training additions are NOT pre2024 historical observations; no temporal generalization claim','historic_test':'ID/hash protection only; no labels or predictions','selection':'candidate only if Macro-F1 gain>=.01, no accuracy decline, no class F1 drop>.02; no automatic app switch'}
(O/'plan.json').write_text(json.dumps(plan,indent=2))
spec=importlib.util.spec_from_file_location('loggedaudit',R/'diagnostics/stage_e_20261028/training_audit.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
def metrics(y,p,labels):return {'accuracy':accuracy_score(y,p),'macro_f1':f1_score(y,p,labels=labels,average='macro'),'per_class':classification_report(y,p,labels=labels,output_dict=True,zero_division=0)}
original=joblib.load(modelpath);model=clone(original)
with threadpool_limits(limits=2):model,record=module.logged_fit(model,train,val,run_id='AUG01',data_version='DATA100_plus18_reviewed_synthetic9groups',train_path=O/'train.csv',validation_path=valpath,run_dir=O/'run',ledger_path=O/'training_ledger.jsonl',metrics_function=metrics,seed=42)
t=time.perf_counter();scores=model.predict_proba(val.narrative);pred=model.classes_[scores.argmax(1)];predict_seconds=time.perf_counter()-t
out=val[['complaint_id','received_date','issue']].copy();out['predicted_issue']=pred
for j in range(7):out[f'class_score_{j}']=scores[:,j]
out.to_csv(O/'validation_predictions.csv',index=False);pd.DataFrame(confusion_matrix(val.issue,pred,labels=model.classes_),index=model.classes_,columns=model.classes_).to_csv(O/'validation_confusion_matrix.csv')
old=pd.read_csv(R/'research/discrimination_20261028/stage_4_experiments/DATA100/validation_predictions.csv',dtype={'complaint_id':str}).set_index('complaint_id').loc[val.complaint_id];assert np.array_equal(old.issue,val.issue)
bm=metrics(val.issue,old.predicted_issue.to_numpy(),list(model.classes_));nm=metrics(val.issue,pred,list(model.classes_));per=[]
for name,rep in [('DATA100',bm),('AUG01',nm)]:
 for label in model.classes_:per.append({'model':name,'issue':label,**rep['per_class'][label]})
pd.DataFrame(per).to_csv(O/'per_class.csv',index=False)
probe=pd.read_csv(R/'research/discrimination_20261028/three_direction_diagnostics/authored_cases_coordinator.csv');p1=original.predict(probe.text);p2=model.predict(probe.text);probe['DATA100_prediction']=p1;probe['AUG01_prediction']=p2;probe['status']='development diagnostic; seed cases partially used for training, not independent evaluation';probe.to_csv(O/'diagnostic_predictions.csv',index=False)
summary=[]
for name,pp in [('DATA100',p1),('AUG01',p2)]:
 for v in [1,5,20]:
  mask=(probe.direction=='B_repetition')&(probe.repeats==v);summary.append({'model':name,'repeats':v,'author_expectation_matches':int((pp[mask]==probe.loc[mask,'expected_author_only']).sum()),'n':int(mask.sum()),'independent':False})
pd.DataFrame(summary).to_csv(O/'diagnostic_summary.csv',index=False)
deltas={l:nm['per_class'][l]['f1-score']-bm['per_class'][l]['f1-score'] for l in model.classes_};gain=nm['macro_f1']-bm['macro_f1'];passed=gain>=.01 and nm['accuracy']>=bm['accuracy'] and min(deltas.values())>=-.02
result={'DATA100':{'accuracy':bm['accuracy'],'macro_f1':bm['macro_f1']},'AUG01':{'accuracy':nm['accuracy'],'macro_f1':nm['macro_f1']},'macro_f1_gain':gain,'passes_gate':bool(passed),'f1_deltas':deltas,'fit_seconds':record['fit_seconds'],'n_iter':record['n_iter_'],'warnings':record['warnings'],'predict_seconds':predict_seconds,'model_bytes':(O/'run/model.joblib').stat().st_size,'validation_predictions_changed':int((pred!=old.predicted_issue.to_numpy()).sum()),'new_fits':1,'app_changed':False,'historical_test_label_or_prediction_access':False}
assert sha(modelpath)==plan['base_model_sha256'] and sha(basepath)==plan['base_train_sha256'] and sha(valpath)==plan['validation_sha256'] and sha(journal[0])==plan['review_journal_sha256']
assert np.array_equal(joblib.load(O/'run/model.joblib').predict(val.narrative.head(3)),pred[:3]);(O/'score_metadata.json').write_text(json.dumps({'labels_in_class_score_order':list(model.classes_),'meaning':'uncalibrated predict_proba; not correctness probability'},indent=2))
(O/'results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
