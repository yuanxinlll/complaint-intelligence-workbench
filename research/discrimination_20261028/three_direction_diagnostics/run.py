from pathlib import Path
import json,sys,hashlib
import pandas as pd,numpy as np,joblib
O=Path(__file__).resolve().parent;R=O.parents[2];sys.path.insert(0,str(R));assert not (O/'summary.json').exists()
labels=['Attempts to collect debt not owed','Communication tactics','Electronic communications','False statements or representation','Threatened to contact someone or share information improperly','Took or threatened to take negative or legal action','Written notification about debt']
seeds=[(0,'This debt belongs to someone else. I have never opened the account.'),(0,'The collector is demanding payment for a debt that I already paid in full.'),(1,'The collector repeatedly calls my workplace after I asked them to stop.'),(1,'The collector calls me repeatedly late at night and uses abusive language.'),(2,'I unsubscribed from the debt collection emails but the company continues sending them.'),(2,'The collector continues sending debt collection text messages after I opted out.'),(3,'The collector falsely claimed to be an attorney when demanding payment.'),(3,'The collector falsely stated the amount I owe and added charges I do not owe.'),(4,'The collector threatened to tell my employer and relatives about my debt.'),(4,'The collector threatened to share my debt details with my coworkers.'),(5,'The collector threatened to sue me and garnish my wages.'),(5,'The collector threatened to seize my property to collect the debt.'),(6,'I requested a written debt validation notice but the collector never sent it.'),(6,'The written debt notice did not explain how I could dispute the debt.')]
phone=' I received a phone call about this account.';cases=[]
for i,(k,text) in enumerate(seeds,1):
 sid=f'S{i:02d}';label=labels[k]
 for variant,n in [('no_neutral_phone',0),('one_neutral_phone',1)]:cases.append({'direction':'A_contrast','seed':sid,'variant':variant,'repeats':n,'expected_author_only':label,'text':text+phone*n})
 for n in [1,5,20]:cases.append({'direction':'B_repetition','seed':sid,'variant':str(n),'repeats':n,'expected_author_only':label,'text':text+phone*n})
 secondary='The collector also keeps calling my workplace after I asked them to stop.' if k!=1 else 'I also have not received a written validation notice.'
 cases.append({'direction':'C_primary_issue','seed':sid,'variant':'mixed','repeats':0,'expected_author_only':label,'text':text+' '+secondary})
 cases.append({'direction':'C_primary_issue','seed':sid,'variant':'explicit_primary','repeats':0,'expected_author_only':label,'text':'My primary complaint is: '+text+' A secondary issue is: '+secondary})
C=pd.DataFrame(cases);assert len(C)==98
C.to_csv(O/'authored_cases_coordinator.csv',index=False)
review=C[['direction','seed','variant','text']].copy();review['human_category']='';review['insufficient_information']='';review['multiple_reasonable_categories']='';review['review_reason']='';review.to_csv(O/'human_review_blank.csv',index=False)
(O/'plan.json').write_text(json.dumps({'directions':['A:14 seeds x no/one neutral phone sentence','B:14 seeds x1/5/20 identical neutral phone sentences','C:14 mixed narratives x implicit/explicit author primary issue'],'models':['v1','DATA100'],'training':False,'labels':'author-declared synthetic expectations, not human-verified truth','selection':'no weighting, thresholds, application changes or parameter search','history_test_access':False},indent=2))
summary=[];outputs=[];transitions=[]
for name,path in [('v1',R/'artifacts/model.joblib'),('DATA100',R/'research/discrimination_20261028/stage_4_experiments/DATA100/model.joblib')]:
 h=hashlib.sha256(path.read_bytes()).hexdigest();m=joblib.load(path);scores=m.predict_proba(C.text);pred=m.classes_[scores.argmax(1)];out=C.copy();out['model']=name;out['prediction']=pred;out['author_expectation_match']=pred==C.expected_author_only
 for j,label in enumerate(m.classes_):out['score_'+str(j)]=scores[:,j]
 outputs.append(out)
 for (direction,variant),g in out.groupby(['direction','variant'],sort=False):summary.append({'model':name,'direction':direction,'variant':variant,'n':len(g),'author_expectation_matches':int(g.author_expectation_match.sum())})
 for sid,g in out.groupby('seed'):
  a=g[g.direction=='A_contrast'].set_index('variant');b=g[g.direction=='B_repetition'].set_index('variant');c=g[g.direction=='C_primary_issue'].set_index('variant')
  transitions.append({'model':name,'seed':sid,'expected_author_only':a.iloc[0].expected_author_only,'neutral_phone_changes_category':bool(a.loc['no_neutral_phone','prediction']!=a.loc['one_neutral_phone','prediction']),'one_to_five_changes_category':bool(b.loc['1','prediction']!=b.loc['5','prediction']),'one_to_twenty_changes_category':bool(b.loc['1','prediction']!=b.loc['20','prediction']),'explicit_primary_changes_category':bool(c.loc['mixed','prediction']!=c.loc['explicit_primary','prediction']),'primary_corrects_author_match':bool(not c.loc['mixed','author_expectation_match'] and c.loc['explicit_primary','author_expectation_match']),'primary_loses_author_match':bool(c.loc['mixed','author_expectation_match'] and not c.loc['explicit_primary','author_expectation_match'])})
 assert hashlib.sha256(path.read_bytes()).hexdigest()==h
pd.concat(outputs).to_csv(O/'all_predictions.csv',index=False);pd.DataFrame(summary).to_csv(O/'summary.csv',index=False);pd.DataFrame(transitions).to_csv(O/'transitions.csv',index=False)
(O/'summary.json').write_text(json.dumps(summary,indent=2));print(pd.DataFrame(summary).to_string(index=False));print(pd.DataFrame(transitions).groupby('model').sum(numeric_only=True).to_string())
