"""Exactly two authorized fixed-D01 data-size fits. No holdout evaluation."""
from pathlib import Path
import os
os.environ.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
import sys,json,hashlib,time,resource,importlib.util,datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
from threadpoolctl import threadpool_limits
R=Path(__file__).resolve().parents[3];O=Path(__file__).resolve().parent;sys.path.insert(0,str(R))
from src.data import clean_text

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encode(x):
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 return str(x)
def save(name,obj):(O/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=encode)+'\n')
def log(s):
 line=f'[{datetime.datetime.now().astimezone().isoformat(timespec="seconds")}] {s}'
 print(line,flush=True)
 with (O/'training.log').open('a') as f:f.write(line+'\n')
def check():
 if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>6_000_000_000:raise RuntimeError('Peak RSS >6GB')
 if fit_total>1800:raise RuntimeError('Total fit time >30min')
def metrics(y,p,labels):return {'accuracy':accuracy_score(y,p),'macro_f1':f1_score(y,p,labels=labels,average='macro',zero_division=0),'per_class':classification_report(y,p,labels=labels,zero_division=0,output_dict=True),'confusion_matrix_labels':labels,'confusion_matrix_true_rows_predicted_columns':confusion_matrix(y,p,labels=labels).tolist()}

if __name__=='__main__':
 started=time.monotonic();fit_total=0.;current=None
 try:
  assert not (O/'training_started.json').exists(),'Do not silently rerun fits'
  template=json.loads((O/'template_summary.json').read_text());plan=json.loads((O/'plan.json').read_text());assert template['data_changed'] is False
  basepath=R/'diagnostics/stage_c_20261028/data/train.csv';valpath=R/'data/processed/validation.csv';newpath=R/'research/discrimination_20261028/stage_4_data/unique_training_candidates.csv'
  for name,path in [('base',basepath),('validation',valpath),('new',newpath)]:assert sha(path)==plan['data_sha256'][name]
  base=pd.read_csv(basepath,dtype=str,keep_default_na=False);val=pd.read_csv(valpath,dtype=str,keep_default_na=False);new=pd.read_csv(newpath,dtype=str,keep_default_na=False)
  assert base.complaint_id.is_unique and new.complaint_id.is_unique
  new['rank']=[hashlib.sha256(f'seed42|{r.complaint_id}|{r.text_hash}'.encode()).hexdigest() for r in new.itertuples()]
  half=new.sort_values(['issue','rank'],kind='stable').groupby('issue',sort=True).head(0) # schema-only, no rows
  half=pd.concat([part.sort_values('rank',kind='stable').head(len(part)//2) for _,part in new.groupby('issue',sort=True)],ignore_index=True)
  assert len(half)==6442 and len(new)==12890 and set(half.complaint_id)<=set(new.complaint_id)
  (O/'data').mkdir(exist_ok=False);frozen={};classrows=[]
  for id,added in [('DATA50',half),('DATA100',new)]:
   train=pd.concat([base,added[base.columns]],ignore_index=True).sort_values(['received_date','complaint_id'],kind='stable').reset_index(drop=True)
   assert train.complaint_id.is_unique and train.text_hash.is_unique
   assert not set(train.complaint_id)&set(val.complaint_id) and not set(train.text_hash)&set(val.text_hash)
   assert train.received_date.le('2024-12-12').all()
   preserved=train.set_index('complaint_id').loc[base.complaint_id,base.columns.drop('complaint_id')].reset_index()
   pd.testing.assert_frame_equal(preserved[base.columns].reset_index(drop=True),base.reset_index(drop=True))
   trainpath=O/f'data/{id}_train.csv';train.to_csv(trainpath,index=False,mode='x');readback=pd.read_csv(trainpath,dtype=str,keep_default_na=False);pd.testing.assert_frame_equal(readback,train)
   added[['complaint_id','text_hash','rank']].sort_values('rank').to_csv(O/f'data/{id}_added_ids.csv',index=False,mode='x')
   frozen[id]={'path':str(trainpath.relative_to(R)),'rows':len(train),'added_rows':len(added),'sha256':sha(trainpath),'added_IDs_sha256':sha(O/f'data/{id}_added_ids.csv'),'date_min':train.received_date.min(),'date_max':train.received_date.max()}
   for label in sorted(train.issue.unique()):classrows.append({'run':id,'issue':label,'base':int(base.issue.eq(label).sum()),'added':int(added.issue.eq(label).sum()),'total':int(train.issue.eq(label).sum())})
  pd.DataFrame(classrows).to_csv(O/'training_counts.csv',index=False,mode='x');save('data_versions.json',frozen)
  modelpath=R/'diagnostics/stage_d_20261028/runs/D01/model.joblib';oldrun=json.loads((modelpath.parent/'run.json').read_text());assert sha(modelpath)==oldrun['model_sha256']
  prototype=joblib.load(modelpath);classifier=prototype.named_steps['classifier'];tfidf=prototype.named_steps['tfidf']
  assert classifier.C==1 and classifier.class_weight=='balanced' and classifier.solver=='lbfgs' and classifier.max_iter==1000 and classifier.tol==.0001
  assert tfidf.ngram_range==(1,2) and tfidf.min_df==2 and tfidf.max_df==.98 and tfidf.max_features==40000 and tfidf.preprocessor is clean_text
  assert tfidf.token_pattern==r'(?u)\b\w+\b'
  save('training_started.json',{'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'script_sha256':sha(__file__),'plan_sha256':sha(O/'plan.json'),'D01_sha256':sha(modelpath),'runs':['DATA50','DATA100'],'no_new_baseline_fit':True,'no_history_test':True})
  spec=importlib.util.spec_from_file_location('audit',R/'diagnostics/stage_e_20261028/training_audit.py');audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
  labels=prototype.classes_.tolist();all_preds={};resultrows=[];perclass=[];pairrows=[];costrows=[]
  validation_flags=set(template['validation_flagged_ids'])
  with threadpool_limits(limits=4):
   for id in ['DATA50','DATA100']:
    current=id;check();trainpath=R/frozen[id]['path'];train=pd.read_csv(trainpath,dtype=str,keep_default_na=False)
    log(f'{id}: frozen {len(train)} training / {len(val)} validation; start single fit.')
    model,record=audit.logged_fit(clone(prototype),train,val,run_id=id,data_version=f'{id}_v2base_plus_AugOct2024_frozen',train_path=trainpath,validation_path=valpath,run_dir=O/id,ledger_path=O/'training_ledger.jsonl',metrics_function=metrics,seed=42)
    fit_total+=record['fit_seconds'];check();t=time.monotonic();scores=model.predict_proba(val.narrative);predict_seconds=time.monotonic()-t
    order=np.argsort(-scores,axis=1,kind='stable');pred=model.classes_[order[:,0]];all_preds[id]=pred
    out=val[['complaint_id','received_date','issue']].copy();out['predicted_issue']=pred;out['second_issue']=model.classes_[order[:,1]];out['top1_score']=scores[np.arange(len(val)),order[:,0]];out['top2_score']=scores[np.arange(len(val)),order[:,1]];out['score_gap']=out.top1_score-out.top2_score
    for j in range(len(labels)):out[f'class_score_{j}']=scores[:,j]
    out.to_csv(O/id/'validation_predictions.csv',index=False,mode='x');save(f'{id}/score_metadata.json',{'labels_in_class_score_order':labels,'output':'uncalibrated predict_proba, not correctness probability','threshold_selected':False})
    met=record['validation_metrics'];save(f'{id}/metrics.json',met)
    np.testing.assert_array_equal(model.predict(val.narrative.head(8)),joblib.load(O/id/'model.joblib').predict(val.narrative.head(8)))
    for label in labels:perclass.append({'run':id,'issue':label,**met['per_class'][label]})
    cm=np.array(met['confusion_matrix_true_rows_predicted_columns']);a=labels.index('Attempts to collect debt not owed')
    for other in ['False statements or representation','Took or threatened to take negative or legal action','Written notification about debt']:
     b=labels.index(other);pairrows.append({'run':id,'A':'Attempts to collect debt not owed','B':other,'A_to_B':int(cm[a,b]),'B_to_A':int(cm[b,a]),'bidirectional_total':int(cm[a,b]+cm[b,a])})
    dummy=np.full(len(val),train.issue.value_counts().index[0]);save(f'{id}/dummy_metrics.json',metrics(val.issue,dummy,labels))
    for flag in [False,True]:
     mask=val.complaint_id.isin(validation_flags).to_numpy()==flag
     save(f'{id}/template_{"flagged" if flag else "unflagged"}_development_metrics.json',{'n':int(mask.sum()),'selection':'fixed template flags only; not separate population or accepted subset','metrics':metrics(val.issue[mask],pred[mask],labels) if mask.any() else None})
    cost={'run':id,'training_rows':len(train),'fit_seconds':record['fit_seconds'],'validation_predict_proba_seconds':predict_seconds,'model_bytes':(O/id/'model.joblib').stat().st_size,'features':len(model.named_steps['tfidf'].vocabulary_),'n_iter':record['n_iter_'],'convergence_warning':record['convergence_warning'],'peak_RSS_bytes_macOS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss};costrows.append(cost)
    resultrows.append({'run':id,'training_rows':len(train),'macro_f1':met['macro_f1'],'accuracy':met['accuracy'],**cost})
    save('partial_results.json',resultrows);log(f'{id} done: Macro-F1={met["macro_f1"]:.6f}, Accuracy={met["accuracy"]:.6f}, fit={record["fit_seconds"]:.2f}s, n_iter={record["n_iter_"]}.')
  # Reuse D01 and best-word D03 saved predictions, never refit or run them on the test set.
  for id in ['D01','D03']:
   df=pd.read_csv(R/f'diagnostics/stage_d_20261028/runs/{id}/validation_predictions.csv',dtype=str).set_index('complaint_id').loc[val.complaint_id]
   assert df.issue.tolist()==val.issue.tolist();pred=df.predicted_issue.to_numpy();all_preds[id]=pred;met=metrics(val.issue,pred,labels)
   resultrows.append({'run':id,'training_rows':6333,'macro_f1':met['macro_f1'],'accuracy':met['accuracy'],'fit_seconds':None,'reference':'reused saved D-stage outputs, zero new fit'})
   for label in labels:perclass.append({'run':id,'issue':label,**met['per_class'][label]})
   cm=np.array(met['confusion_matrix_true_rows_predicted_columns']);a=labels.index('Attempts to collect debt not owed')
   for other in ['False statements or representation','Took or threatened to take negative or legal action','Written notification about debt']:
    b=labels.index(other);pairrows.append({'run':id,'A':'Attempts to collect debt not owed','B':other,'A_to_B':int(cm[a,b]),'B_to_A':int(cm[b,a]),'bidirectional_total':int(cm[a,b]+cm[b,a])})
  pd.DataFrame(resultrows).to_csv(O/'comparison.csv',index=False,mode='x');pd.DataFrame(perclass).to_csv(O/'per_class_metrics.csv',index=False,mode='x');pd.DataFrame(pairrows).to_csv(O/'confusion_pairs.csv',index=False,mode='x');save('costs.json',costrows)
  base_metrics=json.loads((R/'diagnostics/stage_d_20261028/runs/D01/metrics.json').read_text())['metrics'];major='Attempts to collect debt not owed';minor=['Electronic communications','Threatened to contact someone or share information improperly'];selection={}
  for id in ['DATA50','DATA100']:
   met=json.loads((O/id/'metrics.json').read_text());gain=met['macro_f1']-base_metrics['macro_f1'];drops={c:base_metrics['per_class'][c]['f1-score']-met['per_class'][c]['f1-score'] for c in [major]+minor}
   selection[id]={'gain_over_D01':gain,'class_F1_drops':drops,'passes_fixed_gate':gain>=.01 and drops[major]<=.03 and all(drops[c]<=.02 for c in minor),'no_auto_app_switch':True}
  save('selection.json',selection)
  # Two predefined paired differences, fixed seven-class macro F1. Fast count-based calculation.
  label_map={c:i for i,c in enumerate(labels)};truth=np.array([label_map[c] for c in val.issue]);intpred={id:np.array([label_map[c] for c in pred]) for id,pred in all_preds.items()}
  def macro(y,p):
   cm=np.bincount(y*7+p,minlength=49).reshape(7,7);den=cm.sum(0)+cm.sum(1);return np.divide(2*np.diag(cm),den,out=np.zeros(7,dtype=float),where=den>0).mean()
  for id in intpred:assert abs(macro(truth,intpred[id])-f1_score(val.issue,all_preds[id],labels=labels,average='macro',zero_division=0))<1e-12
  rng=np.random.default_rng(42);diffs={id:[] for id in ['DATA50','DATA100']}
  for _ in range(2000):
   ix=rng.integers(0,len(val),len(val));ref=macro(truth[ix],intpred['D01'][ix])
   for id in diffs:diffs[id].append(macro(truth[ix],intpred[id][ix])-ref)
  save('bootstrap.json',{'replicates':2000,'seed':42,'resampling_unit':'individual validation complaint; paired same indices','differences_vs_D01':{id:{'observed':selection[id]['gain_over_D01'],'percentile95':np.percentile(values,[2.5,97.5])} for id,values in diffs.items()},'limitations':'Repeated validation selection bias remains; template-related observations may make intervals optimistic; not independent generalization evidence'})
  current=None
  old=json.loads((O/'protected_before.json').read_text());changed=[n for n,h in old.items() if sha(R/n)!=h];assert not changed
  ledger=[json.loads(x) for x in (O/'training_ledger.jsonl').read_text().splitlines()];assert [e['status'] for e in ledger]==['started','completed','started','completed']
  bytecount=sum(p.stat().st_size for p in O.rglob('*') if p.is_file());assert bytecount<1_000_000_000
  save('checks.json',{'protected_files':len(old),'changed':changed,'new_classifier_fit_started':2,'new_classifier_fits_completed':2,'prior_classifier_fits_repeated':0,'nested_additions_verified':True,'base_rows_unchanged':True,'validation_rows_unchanged':1428,'historical_test_label_or_inference_access':False,'application_changed':False,'data_model_rules_unchanged':True,'new_downloads_installs':0,'model_reload_check':True,'new_artifact_bytes':bytecount,'peak_RSS_bytes_macOS':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'total_fit_seconds':fit_total,'training_phase_elapsed_seconds':time.monotonic()-started,'source_script_sha256':sha(__file__)})
  log('Two authorized fits completed. Stop; application remains v1.')
 except BaseException as e:
  save('training_failure.json',{'run':current,'error_type':type(e).__name__,'error':str(e),'elapsed_seconds':time.monotonic()-started,'fit_total':fit_total});raise
