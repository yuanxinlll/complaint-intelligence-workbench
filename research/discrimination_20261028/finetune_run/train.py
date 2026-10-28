from pathlib import Path
import os
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
import json,time,threading,hashlib,random,resource,traceback
import numpy as np,pandas as pd,torch
from transformers import AutoTokenizer,AutoModelForSequenceClassification
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,f1_score
O=Path(__file__).resolve().parent;R=O.parents[2];P=R/'research/discrimination_20261028/finetune_pilot';assert not (O/'state.json').exists(),'Never overwrite training run'
started=time.monotonic();torch.set_num_threads(4);random.seed(42);np.random.seed(42);torch.manual_seed(42)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
paths={'train':R/'research/discrimination_20261028/stage_4_experiments/data/DATA100_train.csv','validation':R/'data/processed/validation.csv'}
state={'status':'starting','epoch':0,'formal_optimizer_steps':0,'formal_complaints_seen':0,'safety_updates':0,'history':[],'seed':42}
def save(name,obj):
 tmp=O/(name+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(O/name)
def status():
 state.update(elapsed_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss);save('state.json',state)
status();alive=True
# Self RSS API avoids the denied system-wide process enumeration used in the earlier pilot.
def watch():
 while alive:
  peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
  if peak>6000000000 or time.monotonic()-started>28800:
   state.update(status='budget_stopped',reason='6GB RSS or 8h wall budget');status();os._exit(2)
  time.sleep(.5)
threading.Thread(target=watch,daemon=True).start()
try:
 protected={str(p.relative_to(R)):sha(p) for p in R.rglob('*') if p.is_file() and O not in p.parents and '.git' not in p.parts};save('protected_before.json',protected)
 frames={k:pd.read_csv(p,dtype={'complaint_id':str}) for k,p in paths.items()};tr=frames['train'];va=frames['validation'];assert len(tr)==19223 and len(va)==1428
 assert not set(tr.complaint_id)&set(va.complaint_id);assert not set(tr.text_hash)&set(va.text_hash)
 labels=sorted(tr.issue.unique());assert len(labels)==7 and set(labels)==set(va.issue)
 plan={'id':'FT01','data_sha256':{k:sha(p) for k,p in paths.items()},'model_revision':json.loads((P/'download_plan.json').read_text())['revision'],'device':'cpu','threads':4,'epochs_max':3,'seed':42,'optimizer':'AdamW','lr':2e-5,'weight_decay':.01,'microbatch':1,'effective_batch':16,'float':'float32','class_weights':'N/(7*n_class), training labels only','loss':'sum(weight_i * CE_i)/sum(weight_i) over effective batch','clip_grad_norm':1.0,'max_length':512,'long_text':'first255+last255 content tokens plus CLS SEP','checkpoint_selection':'max validation Macro-F1 each epoch; tie earliest','validation_evaluations_max':3,'historical_test_access':False,'app_switch':False,'budgets':{'seconds':28800,'peak_rss_bytes':6000000000}}
 save('plan.json',plan);save('environment.json',{'torch':torch.__version__,'python':os.sys.version,'script_sha256':sha(Path(__file__))})
 tok=AutoTokenizer.from_pretrained(P/'model',local_files_only=True);encoded={}
 for split,f in frames.items():
  result=[]
  for i in range(0,len(f),128):
   for ids in tok(f.narrative.fillna('').iloc[i:i+128].tolist(),add_special_tokens=False,truncation=False,verbose=False)['input_ids']:
    ids=ids if len(ids)<=510 else ids[:255]+ids[-255:];result.append([tok.cls_token_id]+ids+[tok.sep_token_id])
  encoded[split]=result
 targets=torch.tensor([labels.index(x) for x in tr.issue]);weights=torch.tensor([len(tr)/(7*int((tr.issue==x).sum())) for x in labels],dtype=torch.float32)
 def inputs(ids):return {'input_ids':torch.tensor([ids]),'attention_mask':torch.ones((1,len(ids)),dtype=torch.long)}
 def fresh():return AutoModelForSequenceClassification.from_pretrained(P/'model',num_labels=7,id2label=dict(enumerate(labels)),label2id={x:i for i,x in enumerate(labels)},local_files_only=True)
 # Verify accumulation against one batched weighted CE using synthetic logits only.
 q=torch.randn(5,7,requires_grad=True);y=torch.tensor([0,1,2,3,4]);ce=torch.nn.functional.cross_entropy(q,y,weight=weights);g=torch.autograd.grad(ce,q)[0];q2=q.detach().clone().requires_grad_();loss=sum(torch.nn.functional.cross_entropy(q2[i:i+1],y[i:i+1],reduction='sum')*weights[y[i]] for i in range(5))/weights[y].sum();g2=torch.autograd.grad(loss,q2)[0];assert torch.allclose(g,g2,atol=1e-6);save('gradient_check.json',{'weighted_effective_batch_equivalence':True})
 state['status']='longest_input_safety';status();torch.manual_seed(42);m=fresh();m.train();opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=.01)
 ix=max(range(len(tr)),key=lambda i:len(encoded['train'][i]));t=time.monotonic();out=m(**inputs(encoded['train'][ix])).logits;loss=torch.nn.functional.cross_entropy(out,targets[ix:ix+1]);loss.backward();assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None);opt.step();state['safety_updates']=1
 save('safety.json',{'training_id':tr.complaint_id.iloc[ix],'sequence_length':len(encoded['train'][ix]),'seconds':time.monotonic()-t,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'updates':1,'weights_discarded':True});del opt,m,out,loss
 import gc;gc.collect();torch.manual_seed(42);m=fresh();m.train();opt=torch.optim.AdamW(m.parameters(),lr=2e-5,weight_decay=.01);rng=np.random.default_rng(42);best=-1;fit_total=0
 for epoch in range(1,4):
  state.update(status='training',epoch=epoch,epoch_seen=0);status();m.train();order=rng.permutation(len(tr));epoch_start=time.monotonic();loss_sum=0
  for offset in range(0,len(order),16):
   idx=order[offset:offset+16];denom=weights[targets[idx]].sum();opt.zero_grad(set_to_none=True)
   for j in idx:
    logits=m(**inputs(encoded['train'][j])).logits
    item=torch.nn.functional.cross_entropy(logits,targets[j:j+1],reduction='sum')*weights[targets[j]]/denom
    assert torch.isfinite(item);item.backward();loss_sum+=float(item.detach())
   torch.nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True);opt.step();state['formal_optimizer_steps']+=1;state['formal_complaints_seen']+=len(idx);state['epoch_seen']+=len(idx)
   if state['formal_optimizer_steps']%10==0:status()
   if state['formal_optimizer_steps']%100==0:print('epoch',epoch,'seen',state['epoch_seen'],'elapsed',round(time.monotonic()-started),flush=True)
  fit_seconds=time.monotonic()-epoch_start;fit_total+=fit_seconds;state['status']='validation';status();m.eval();scores=[];t=time.monotonic()
  with torch.inference_mode():
   for ids in encoded['validation']:scores.append(m(**inputs(ids)).logits.softmax(-1).numpy()[0])
  predict_seconds=time.monotonic()-t;scores=np.array(scores);assert np.isfinite(scores).all();pred=np.array(labels)[scores.argmax(1)];rep=classification_report(va.issue,pred,labels=labels,output_dict=True,zero_division=0);macro=f1_score(va.issue,pred,labels=labels,average='macro');acc=accuracy_score(va.issue,pred)
  E=O/f'epoch_{epoch}';E.mkdir();df=va[['complaint_id','received_date','issue']].copy();df['predicted_issue']=pred
  for c in range(7):df[f'class_score_{c}']=scores[:,c]
  df.to_csv(E/'validation_predictions.csv',index=False);(E/'metrics.json').write_text(json.dumps({'accuracy':acc,'macro_f1':macro,'classification_report':rep},indent=2));pd.DataFrame(confusion_matrix(va.issue,pred,labels=labels),index=labels,columns=labels).to_csv(E/'confusion_matrix.csv')
  record={'epoch':epoch,'accuracy':acc,'macro_f1':macro,'training_seconds':fit_seconds,'validation_seconds':predict_seconds,'mean_effective_batch_loss':loss_sum/((len(order)+15)//16)};state['history'].append(record)
  if macro>best:
   best=macro;state['best_epoch']=epoch;m.save_pretrained(O/'best_model');tok.save_pretrained(O/'best_model')
  status();print('EPOCH_RESULT',json.dumps(record),flush=True)
 state.update(status='completed',fit_total_seconds=fit_total,model_size_bytes=sum(p.stat().st_size for p in (O/'best_model').rglob('*') if p.is_file()));status();alive=False
 save('score_metadata.json',{'labels_in_class_score_order':labels,'meaning':'uncalibrated softmax; not correctness probability'})
 changed=[n for n,h in protected.items() if sha(R/n)!=h];save('checks.json',{'changed_existing_files':changed,'safety_updates':1,'formal_training_runs':1,'epochs':3,'historical_test_access':False,'application_changed':False});assert not changed
except BaseException as e:
 alive=False;state.update(status='failed',error=repr(e),traceback=traceback.format_exc());status();raise
