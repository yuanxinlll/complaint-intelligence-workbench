"""Local versioned model loading; no downloads, network calls or online training."""
from pathlib import Path
import hashlib,json,threading,os
import numpy as np
from src.data import ROOT

def registry():
    path = 'cloud_demo/model_profiles.json' if os.environ.get('CIW_PUBLIC_DEMO') == '1' else 'config/model_profiles.json'
    return json.loads((ROOT/path).read_text())

def verify(path,expected):
    if not path.is_file():raise ValueError('Local model files are missing. See README; no download was started.')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected:raise ValueError('Model checksum mismatch; use a trusted matching model version.')

class DistilBertClassifier:
    def __init__(self,path):
        import torch
        from transformers import AutoTokenizer,AutoModelForSequenceClassification
        self.torch=torch;torch.set_num_threads(4);self.lock=threading.Lock()
        self.tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
        self.model=AutoModelForSequenceClassification.from_pretrained(path,local_files_only=True).to('cpu').eval()
        self.classes_=np.array([self.model.config.id2label[i] for i in range(self.model.config.num_labels)])
        if len(self.classes_)!=7:raise ValueError('Expected seven output categories.')
    def content_ids(self,text):
        if not isinstance(text,str) or not text.strip():raise ValueError('Enter a nonempty complaint narrative.')
        if len(text)>50000:raise ValueError('Narrative exceeds 50,000 characters.')
        return self.tokenizer(text,add_special_tokens=False,truncation=False,verbose=False)['input_ids']
    def input_details(self,text):
        n=len(self.content_ids(text));return {'content_tokens':n,'truncated':n>510,'discarded_middle_tokens':max(0,n-510),'strategy':'first 255 + last 255 content tokens if over 510; two special tokens'}
    def encode(self,text):
        ids=self.content_ids(text);ids=ids if len(ids)<=510 else ids[:255]+ids[-255:]
        return [self.tokenizer.cls_token_id]+ids+[self.tokenizer.sep_token_id]
    def predict_proba(self,texts):
        texts=list(texts)
        if not texts:return np.empty((0,7))
        values=[]
        with self.lock,self.torch.inference_mode():
            for start in range(0,len(texts),2):
                seq=[self.encode(t) for t in texts[start:start+2]];size=max(map(len,seq));ids=[x+[self.tokenizer.pad_token_id]*(size-len(x)) for x in seq];masks=[[1]*len(x)+[0]*(size-len(x)) for x in seq]
                logits=self.model(input_ids=self.torch.tensor(ids),attention_mask=self.torch.tensor(masks)).logits
                values.append(logits.softmax(-1).cpu().numpy())
        out=np.concatenate(values)
        if not np.isfinite(out).all():raise ValueError('Non-finite model output; no suggestion was returned.')
        return out
    def predict(self,texts):return self.classes_[self.predict_proba(texts).argmax(axis=1)]

def load_profile(model_id):
    profiles=registry()['profiles']
    if model_id not in profiles:raise ValueError('Unknown model version.')
    info=profiles[model_id];path=ROOT/info['path']
    if info['kind']=='distilbert':
        for filename,digest in info['sha256'].items():verify(path/filename,digest)
        try:model=DistilBertClassifier(path)
        except ImportError as e:raise ValueError('The semantic runtime is unavailable. Use the documented isolated environment or select DATA100.') from e
    else:
        import joblib
        verify(path,info['sha256']);model=joblib.load(path)
    rule={'mode':'review_all','threshold':None,'model_version':info['version'],'reason':'All suggestions require human confirmation; no automatic acceptance threshold selected.'}
    return model,rule,info
