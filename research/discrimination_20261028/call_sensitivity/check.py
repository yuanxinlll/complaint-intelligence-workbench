"""One fixed post-hoc feature attenuation trial, not classifier retraining."""
from pathlib import Path
import json,time,hashlib,copy,sys
import numpy as np,pandas as pd,joblib
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
O=Path(__file__).resolve().parent;R=O.parents[2];sys.path.insert(0,str(R));assert not (O/'results.json').exists(),'Preserve this trial'
words={'call','calls','called','calling','phone','phones','phoned','phoning','telephone','telephones'}
paths={'v1':R/'artifacts/model.joblib','DATA100':R/'research/discrimination_20261028/stage_4_experiments/DATA100/model.joblib'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
plan={'operation':'Multiply existing LR coefficients by0.5 for every TF-IDF feature containing an exact token in the fixed list. Includes positive and negative coefficients; intercept unchanged. No training, text deletion or feature refit.','tokens':sorted(words),'factor':.5,'variants_per_model':1,'models':{k:sha(p) for k,p in paths.items()},'evaluation':'complete existing validation1428; no historical test','selection_rule':'recommend only if Macro-F1 gain>=.01 vs corresponding original, Accuracy not lower, no class F1 drop>.02; otherwise do not adopt','application_change':False}
(O/'plan.json').write_text(json.dumps(plan,indent=2))
val=pd.read_csv(R/'data/processed/validation.csv',dtype={'complaint_id':str});assert len(val)==1428
cases={'not_owed':'I do not owe this debt. This account belongs to another person.','written_notice':'I requested a written validation notice about the debt but never received one.','legal_threat':'The collector threatened to sue me and garnish my wages.','communication':'The collector keeps calling my workplace after I asked them to stop.'}
synthetic=[]
for name,text in cases.items():
 for n in [0,1,5,20]:synthetic.append({'case':name,'repetitions':n,'text':text+(' I received a phone call.'*n)})
S=pd.DataFrame(synthetic);S.to_csv(O/'synthetic_inputs.csv',index=False)
results=[];per=[];features=[];trials=[]
for name,path in paths.items():
 m=joblib.load(path);candidate=copy.deepcopy(m);tf=m.named_steps['tfidf'];clf=m.named_steps['classifier'];names=tf.get_feature_names_out();mask=np.array([bool(set(str(x).split())&words) for x in names]);idx=np.where(mask)[0];assert len(idx)>0
 candidate.named_steps['classifier'].coef_[:,idx]*=.5
 ci=list(clf.classes_).index('Communication tactics')
 for j in idx:features.append({'model':name,'feature':names[j],'communication_coefficient_before':float(clf.coef_[ci,j]),'communication_coefficient_after':float(candidate.named_steps['classifier'].coef_[ci,j])})
 variants={'original':m,'attenuated_0.5':candidate};classreports={}
 for kind,model in variants.items():
  t=time.perf_counter();score=model.predict_proba(val.narrative);pred=model.classes_[score.argmax(1)];secs=time.perf_counter()-t
  rep=classification_report(val.issue,pred,labels=model.classes_,output_dict=True,zero_division=0);classreports[kind]=rep
  results.append({'model':name,'variant':kind,'accuracy':accuracy_score(val.issue,pred),'macro_f1':f1_score(val.issue,pred,labels=model.classes_,average='macro'),'inference_seconds':secs,'affected_features':int(mask.sum())})
  for label in model.classes_:per.append({'model':name,'variant':kind,'issue':label,**rep[label]})
  dest=O/(name+'_'+kind);dest.mkdir();out=val[['complaint_id','issue']].copy();out['predicted_issue']=pred
  for j in range(7):out[f'class_score_{j}']=score[:,j]
  out.to_csv(dest/'validation_predictions.csv',index=False);pd.DataFrame(confusion_matrix(val.issue,pred,labels=model.classes_),index=model.classes_,columns=model.classes_).to_csv(dest/'confusion_matrix.csv')
  ss=model.predict_proba(S.text)
  for i,row in S.iterrows():trials.append({'model':name,'variant':kind,'case':row['case'],'repetitions':int(row.repetitions),'prediction':str(model.classes_[ss[i].argmax()]),'communication_score':float(ss[i,ci])})
 # retain the trial as a separate model, never overwrite production artifacts
 joblib.dump(candidate,O/(name+'_attenuated_0.5.joblib'))
 before=next(x for x in results if x['model']==name and x['variant']=='original');after=results[-1];drops={l:classreports['attenuated_0.5'][l]['f1-score']-classreports['original'][l]['f1-score'] for l in model.classes_}
 passed=after['macro_f1']-before['macro_f1']>=.01 and after['accuracy']>=before['accuracy'] and min(drops.values())>=-.02
 (O/(name+'_decision.json')).write_text(json.dumps({'gate_passed':bool(passed),'macro_f1_delta':after['macro_f1']-before['macro_f1'],'accuracy_delta':after['accuracy']-before['accuracy'],'class_f1_deltas':drops,'application_switched':False},indent=2))
assert all(sha(paths[k])==h for k,h in plan['models'].items())
pd.DataFrame(results).to_csv(O/'comparison.csv',index=False);pd.DataFrame(per).to_csv(O/'per_class.csv',index=False);pd.DataFrame(features).to_csv(O/'feature_coefficients.csv',index=False);pd.DataFrame(trials).to_csv(O/'synthetic_results.csv',index=False)
(O/'results.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
