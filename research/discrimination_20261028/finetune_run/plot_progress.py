"""Generate shareable plots from saved records only; never run inference."""
from pathlib import Path
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
O=Path(__file__).resolve().parent;D=O/'figures';D.mkdir(exist_ok=True)
s=json.loads((O/'state.json').read_text());history=s.get('history',[]);stamp=datetime.now().strftime('%Y-%m-%d %H:%M local')
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
fig,ax=plt.subplots(1,2,figsize=(12,4.8));fig.suptitle('DistilBERT fine-tuning | recorded progress',fontsize=17,fontweight='bold')
epoch=s.get('epoch',0);seen=s.get('epoch_seen',0);fraction=min(seen/19223,1)
ax[0].barh([0],[1],color='#e6eaf0',height=.35);ax[0].barh([0],[fraction],color='#167c80',height=.35);ax[0].set(xlim=(0,1),yticks=[],xticks=[0,.25,.5,.75,1],xticklabels=['0%','25%','50%','75%','100%']);ax[0].set_title(f'Epoch {epoch} / 3: {fraction:.1%} training records')
ax[0].text(0,-.6,f'{seen:,} / 19,223 records in this epoch\n{s.get("formal_optimizer_steps",0):,} optimizer updates overall\nElapsed: {s.get("elapsed_seconds",0)/60:.1f} minutes\nPeak process memory: {s.get("peak_rss_bytes",0)/1e9:.2f} GB',va='top',linespacing=1.6);ax[0].set_ylim(-1.8,.6)
if history:
 x=[r['epoch'] for r in history];ax[1].plot(x,[r['macro_f1'] for r in history],'o-',label='Validation Macro-F1',color='#167c80');ax[1].plot(x,[r['accuracy'] for r in history],'s-',label='Validation accuracy',color='#dc8239');ax[1].set(xlabel='Completed epoch',ylabel='Score',ylim=(0,1),xticks=[1,2,3]);ax[1].legend();ax[1].set_title('Completed full-validation results')
else:
 ax[1].axis('off');ax[1].text(.05,.75,'Validation results pending',fontsize=16,fontweight='bold');ax[1].text(.05,.55,'No completed epoch evaluation is saved yet.\nA score curve requires recorded results.\nProgress is not evidence of accuracy improvement.',linespacing=1.7)
fig.text(.02,.025,f'Snapshot: {stamp} | CPU run | Reused development validation, not an independent final test.',fontsize=9,color='#56616f');fig.tight_layout(rect=(0,.08,1,.92));fig.savefig(D/'training_progress.png',dpi=160);plt.close(fig)
names=['D01\n6,333 rows','SEM01\n6,333 rows','DATA50\n12,775 rows','DATA100\n19,223 rows','ENS01\nmixed data'];macro=[.3931047619,.3833538665,.4217308081,.4409106465,.4383941805];acc=[.5196078431,.4425770308,.5315126050,.5308123249,.5147058824]
fig,axes=plt.subplots(1,2,figsize=(12,5.1));colors=['#94a4b5','#94a4b5','#8fc5c0','#167c80','#d6a16b']
for a,vals,title in zip(axes,[macro,acc],['Macro-F1 (primary metric)','Accuracy']):
 a.tick_params(axis='x',labelsize=9);a.bar(names,vals,color=colors,width=.65);a.set_ylim(0,.65);a.set_title(title);a.set_ylabel('Development validation score');a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
 for i,v in enumerate(vals):a.text(i,v+.012,f'{v:.4f}',ha='center',fontsize=10)
fig.suptitle('Completed experiments | same 1,428 validation records',fontsize=16,fontweight='bold');fig.text(.02,.03,'Different training versions are shown explicitly. These are experiment comparisons, not an epoch learning curve.\nDistilBERT is not plotted until a completed evaluation is available. Validation has been repeatedly used.',fontsize=9,color='#56616f');fig.tight_layout(rect=(0,.1,1,.92));fig.savefig(D/'experiment_comparison.png',dpi=160);plt.close(fig)
(D/'snapshot.json').write_text(json.dumps({'state':s,'created_local':stamp,'curves_use_saved_epochs_only':True},indent=2));print('Saved training_progress.png and experiment_comparison.png')
