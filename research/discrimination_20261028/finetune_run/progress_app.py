"""Read-only local progress view, independent of the classification app."""
from pathlib import Path
from datetime import datetime,timedelta
import json,time
import pandas as pd
import streamlit as st
O=Path(__file__).resolve().parent
st.set_page_config(page_title='Training Progress | Complaint Intelligence',page_icon='📈',layout='wide')
st.title('Training Progress')
st.caption('Complaint Intelligence Workbench · Local CPU fine-tuning · Read-only monitor')
@st.fragment(run_every='5s')
def dashboard():
 try:s=json.loads((O/'state.json').read_text())
 except (OSError,ValueError):st.warning('Waiting for a readable training status…');return
 audit_path = O / 'completion_audit.json'
 audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
 audited_complete = audit.get('training_and_evaluation_complete', False) and audit.get('source_data_old_models_unchanged', False)
 status = 'completed' if audited_complete else s.get('status', 'unknown')
 if audited_complete and s.get('status') != 'completed':
  st.info('Training and all three evaluations completed. A final preservation assertion flagged only two Python cache files; source, data and original model hashes were verified unchanged. Original error logs are retained.')
 st.subheader({'training':'Training in progress','validation':'Evaluating the complete validation set','completed':'Training completed','failed':'Training stopped with an error','budget_stopped':'Resource budget reached','longest_input_safety':'Checking the longest input'}.get(status,status.replace('_',' ').title()))
 epoch=s.get('epoch',0);seen=s.get('formal_complaints_seen',0);elapsed=s.get('elapsed_seconds',0);n=19223;total=n*3
 cols=st.columns(4)
 cols[0].metric('Epoch',f'{epoch} / 3');cols[1].metric('Training records processed',f'{seen:,} / {total:,}');cols[2].metric('Elapsed time',str(timedelta(seconds=int(elapsed))));cols[3].metric('Peak process memory',f"{s.get('peak_rss_bytes',0)/1e9:.2f} GB")
 st.write('Current epoch');st.progress(min(s.get('epoch_seen',0)/n,1),text=f"{s.get('epoch_seen',0):,} / {n:,} records")
 st.write('Training passes completed');st.progress(min(seen/total,1),text=f'{seen/total:.1%} of the maximum three training passes')
 st.caption('Training progress excludes validation and checkpoint saving. Each epoch uses the same 19,223 training records; these are not additional unique complaints.')
 now=time.time();age=max(0,now-(O/'state.json').stat().st_mtime)
 if age>90 and status not in ('completed','failed','budget_stopped'):st.warning(f'No status update for {int(age)} seconds. Check the log; this does not prove training is still running.')
 if seen>0 and status not in ('completed','failed','budget_stopped'):
  train_seconds=sum(x.get('training_seconds',0) for x in s.get('history',[]));epoch_seconds=max(0,elapsed-17.3-sum(x.get('validation_seconds',0) for x in s.get('history',[])))
  rate=seen/max(epoch_seconds,1);remaining=max(0,total-seen)/max(rate,.01);eta=datetime.now()+timedelta(seconds=remaining)
  st.info(f'Rough remaining training time: {timedelta(seconds=int(remaining))} · Estimated training finish: {eta:%H:%M} local time. Validation and saving add time; this is not a guarantee.')
 st.caption(f'Status updated {int(age)} seconds ago · Refreshes every 5 seconds · CPU, 4 threads · Deadline: 19:00 local time')
 history=s.get('history',[])
 st.subheader('Validation results by epoch')
 if history:
  df=pd.DataFrame(history);st.dataframe(df,hide_index=True,width='stretch');st.caption(f"Selected checkpoint: epoch {s.get('best_epoch','pending')}, based on validation Macro-F1.")
 else:st.info('No new accuracy score yet. The first result appears after a full epoch and validation evaluation.')
 if history:
  st.subheader("Validation learning curve")
  curve = pd.DataFrame(history).set_index("epoch")[["macro_f1", "accuracy"]]
  st.line_chart(curve, x_label="Completed epoch", y_label="Development validation score")
  st.caption("Only completed epoch evaluations are plotted. One point does not establish a trend; training loss is not accuracy.")
 with st.expander("Earlier experiment comparison"):
  figure = O / "figures/experiment_comparison.png"
  if figure.exists():
   st.image(str(figure))
   st.download_button("Download experiment comparison", figure.read_bytes(), "experiment_comparison.png", "image/png")
 st.caption('Reference DATA100: Macro-F1 0.4409 · Accuracy 53.08%. Validation has been used repeatedly; results are development evidence, not an independent final test.')
 if s.get('error') and not audited_complete:st.error(s['error'])
 with st.expander('Recent technical log'):
  p=O/'training.log'
  if p.exists():
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-12000));content=f.read().decode('utf-8',errors='replace')
   st.code(content[-9000:],language='text')
 if audited_complete:
  st.caption('Final review: epoch 3 has the highest Macro-F1 (.4629), but Accuracy (52.24%) is below DATA100 (53.08%). Two small classes improve; other class F1 scores decrease. No automatic model switch.')
 st.caption('The classification app still uses its previous model. This page does not train, change weights, or switch the app model.')
dashboard()
