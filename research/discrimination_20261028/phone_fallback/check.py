from pathlib import Path
import sys,json,hashlib,copy
import numpy as np,pandas as pd,joblib
from sklearn.metrics import classification_report,accuracy_score,f1_score,confusion_matrix
O=Path(__file__).resolve().parent;R=O.parents[2];sys.path.insert(0,str(R));assert not (O/'results.json').exists()
words={'call','calls','called','calling','caller','phone','phones','phoned','phoning','telephone','telephones'}
plan={'hypothesis':'phone features as fallback only','operation':'zero every-class coefficient of exact-token phone features; if resulting top2 probability gap>=.10 use that prediction, otherwise keep original prediction','gap':.10,'threshold_origin':'exploratory fixed assumption before outcomes; not learned or evidence that non-phone information is sufficient','scope':'v1 and DATA100; one rule only','no_fit':True,'no_app_switch':True,'evaluation':'full reused validation1428 plus synthetic diagnostic set; no historical test'}
(O/'plan.json').write_text(json.dumps(plan,indent=2))
contexts=[('not_owed','Attempts to collect debt not owed','I do not owe this debt. The account belongs to a different person.'),('notice','Written notification about debt','I have not received the required written debt validation notice despite requesting it.'),('legal','Took or threatened to take negative or legal action','The collector threatened to sue me and garnish my wages for this debt.'),('harassment','Communication tactics','The collector repeatedly calls my workplace after I asked them to stop. The constant calls are the problem.'),('no_legal_threat','Communication tactics','The collector did not threaten to sue me. My complaint is repeated calls at work after I asked them to stop.'),('no_call_complaint','Written notification about debt','I am not complaining about the phone contact. My complaint is that the collector never sent a written validation notice.')]
rows=[]
for key,label,text in contexts:
 for n in [0,1,5,20]:rows.append({'case':key,'repeats':n,'intended_category':label,'text':text+' I received a phone call.'*n})
S=pd.DataFrame(rows);S.to_csv(O/'synthetic_cases.csv',index=False)
val=pd.read_csv(R/'data/processed/validation.csv',dtype={'complaint_id':str});assert len(val)==1428;summary=[];details=[];per=[];changes=[]
for name,path in [('v1',R/'artifacts/model.joblib'),('DATA100',R/'research/discrimination_20261028/stage_4_experiments/DATA100/model.joblib')]:
 oldhash=hashlib.sha256(path.read_bytes()).hexdigest();m=joblib.load(path);nonphone=copy.deepcopy(m);features=m.named_steps['tfidf'].get_feature_names_out();mask=np.array([bool(set(x.split())&words) for x in features]);nonphone.named_steps['classifier'].coef_[:,mask]=0
 def compare(texts):
  a=m.predict_proba(texts);b=nonphone.predict_proba(texts);ordered=np.sort(b,axis=1);gap=ordered[:,-1]-ordered[:,-2];use=gap>=.10;score=np.where(use[:,None],b,a);return m.classes_[a.argmax(1)],m.classes_[score.argmax(1)],score,use,gap
 original,rule,scores,use,gap=compare(val.narrative)
 for kind,pred in [('original',original),('phone_fallback',rule)]:
  rep=classification_report(val.issue,pred,labels=m.classes_,output_dict=True,zero_division=0);summary.append({'model':name,'variant':kind,'accuracy':accuracy_score(val.issue,pred),'macro_f1':f1_score(val.issue,pred,labels=m.classes_,average='macro')})
  for label in m.classes_:per.append({'model':name,'variant':kind,'issue':label,**rep[label]})
  pd.DataFrame(confusion_matrix(val.issue,pred,labels=m.classes_),index=m.classes_,columns=m.classes_).to_csv(O/(name+'_'+kind+'_confusion.csv'))
 out=val[['complaint_id','issue']].copy();out['original_prediction']=original;out['fallback_prediction']=rule;out['nonphone_selected']=use;out['nonphone_gap']=gap
 for c in range(7):out[f'class_score_{c}']=scores[:,c]
 out.to_csv(O/(name+'_validation_predictions.csv'),index=False)
 changes.append({'model':name,'nonphone_selected':int(use.sum()),'changed_predictions':int((original!=rule).sum()),'corrected':int(((original!=val.issue)&(rule==val.issue)).sum()),'lost_correct':int(((original==val.issue)&(rule!=val.issue)).sum())})
 a,b,_,u,g=compare(S.text)
 for i,row in S.iterrows():details.append({'model':name,**row.to_dict(),'original_prediction':a[i],'fallback_prediction':b[i],'nonphone_selected':bool(u[i]),'nonphone_gap':float(g[i])})
 assert hashlib.sha256(path.read_bytes()).hexdigest()==oldhash
pd.DataFrame(summary).to_csv(O/'comparison.csv',index=False);pd.DataFrame(per).to_csv(O/'per_class.csv',index=False);pd.DataFrame(details).to_csv(O/'synthetic_results.csv',index=False)
(O/'results.json').write_text(json.dumps({'metrics':summary,'changes':changes,'classes':list(m.classes_),'originals_unchanged':True},indent=2));print((O/'results.json').read_text())
