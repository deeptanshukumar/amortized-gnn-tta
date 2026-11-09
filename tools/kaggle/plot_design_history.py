"""Plot saved validation histories only; no fitting or final-label access."""
import argparse,csv,json,math
from pathlib import Path
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--download',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(1)
proof=json.loads((a.download/'LOCAL_STUDY_VERIFICATION.json').read_text())
if proof.get('status')!='complete' or proof.get('mode')!='design':raise ValueError('Verified input/site study required')
study=a.download/proof['study_relative_path'];rows=[];fits=[]
for fit in proof['fits']:
    saved=torch.load(study/fit['name']/'generator.pt',map_location='cpu',weights_only=True)
    history=saved['history']
    if len(history)!=5 or [r['epoch'] for r in history]!=[1,2,3,4,5]:raise ValueError('Unexpected saved training history')
    for row in history:
        if not math.isfinite(row['val_gain']):raise ValueError('Nonfinite validation history')
        rows.append(dict(fit=fit['name'],conditioning=fit['conditioning'],update=fit['update'],**row))
    if fit['selected_epoch']==0 and max(r['val_gain'] for r in history)>0:raise ValueError('History contradicts checkpoint selection')
    fits.append((fit,history))
a.output_dir.mkdir(parents=True,exist_ok=True)
with (a.output_dir/'TRAINING_HISTORY.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,1,figsize=(9,7),sharex=True)
colors={'A':'#0072B2','B':'#D55E00','C':'#009E73','D':'#CC79A7'}
for ax,kind in zip(axes,('lora','head')):
    for fit,history in fits:
        if fit['update']!=kind:continue
        ax.plot([0]+[r['epoch'] for r in history],[0]+[100*r['val_gain'] for r in history],marker='o',markersize=4,label=fit['conditioning'],color=colors[fit['conditioning']])
    ax.axhline(0,color='#333333',linewidth=1,linestyle='--')
    ax.set_title('LoRA at both layers' if kind=='lora' else 'Head-only follow-up',loc='left',fontsize=11)
    ax.set_ylabel('Validation gain (percentage points)')
    ax.grid(axis='y',alpha=.2);ax.legend(title='Inputs',ncol=4,loc='lower right')
axes[-1].set_xlabel('Generator training epoch (0 = zero-update checkpoint)')
axes[-1].set_xticks(range(6))
fig.suptitle('Formal GCN: trained epochs versus the frozen model',fontsize=14,y=.98)
fig.text(.08,.015,'Validation: six source checkpoints × seven cases. Separate vertical scales.\nAll six fits selected epoch 0; final test labels were not used.',fontsize=9)
fig.tight_layout(rect=(0,.07,1,.95))
for extension in ('png','svg'):fig.savefig(a.output_dir/('VALIDATION_HISTORY.'+extension),dpi=180)
plt.close(fig)
print(json.dumps(dict(training_rows=len(rows),figures=['VALIDATION_HISTORY.png','VALIDATION_HISTORY.svg'],final_test_scored=False)))
