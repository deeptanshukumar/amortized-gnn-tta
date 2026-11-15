"""CPU-only compact Experiment 2 report. Reads metrics; no training/test scoring."""
from pathlib import Path
import modal
app=modal.App('afml-experiment2-simple-report')
volume=modal.Volume.from_name('afml-hypothesis-results',create_if_missing=False)
image=(modal.Image.debian_slim(python_version='3.13')
       .pip_install('numpy==2.1.3','pandas==2.3.3','matplotlib==3.10.8')
       .env({'MPLBACKEND':'Agg','OPENBLAS_NUM_THREADS':'1'})
       .add_local_file(str(Path(__file__).resolve()),'/experiment/modal_report_experiment2.py'))
ARCHES=['gcn','graphsage','gat'];INPUTS={'gcn':'C','graphsage':'C','gat':'A'}
NAMES={'gcn':'GCN','graphsage':'GraphSAGE','gat':'GAT'};SEEDS=[202801,202802,202803]


def digest(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(root,source,out):
    import json,zipfile
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);source=Path(source);out=Path(out)
    # Refuse partial campaigns before creating the report output.
    for a in ARCHES:
        if not (source/a/'COMPLETE.json').is_file():raise ValueError(f'Backbone incomplete: {a}')
    out.mkdir(parents=True,exist_ok=False)
    rows=[];curves=[];selected_cases=[]
    for arch in ARCHES:
        plan=json.loads((source/arch/'EXPERIMENT_PLAN.json').read_text())
        if plan['architecture']!=arch or plan['input']!=INPUTS[arch] or plan['seeds']!=SEEDS:raise ValueError('Experiment plan mismatch')
        for seed in SEEDS:
            reference=plan['references'][str(seed)]
            # Resolve /results paths relative to root, also allowing local fixtures.
            original=Path(reference['folder'])
            ref=root/original.relative_to('/results')
            if digest(ref/'RESULT.json')!=reference['result_sha256'] or digest(ref/'history.csv')!=reference['history_sha256']:
                raise ValueError('LoRA reference changed since Experiment 2 training')
            head=source/arch/f'seed_{seed}'/f'{INPUTS[arch]}-head'
            records={};values={}
            for method,folder in [('Head',head),('LoRA',ref)]:
                r=json.loads((folder/'RESULT.json').read_text());h=pd.read_csv(folder/'history.csv');v=pd.read_csv(folder/'validation_cases.csv')
                if r['seed']!=seed or r['conditioning']!=INPUTS[arch] or r['update']!=('head' if method=='Head' else 'lora') or r['loss_kind']!='kl':
                    raise ValueError(f'Wrong method/seed/input: {folder}')
                if r.get('architecture',arch)!=arch:raise ValueError('Wrong architecture')
                if len(v)!=42*(len(h)+1) or v.duplicated(['epoch','model_id','environment_id']).any():raise ValueError('Incomplete validation history')
                best=int(h.sort_values(['val_gain','epoch'],ascending=[False,True]).iloc[0].epoch)
                original_epoch=int(r['selected_epoch'])
                for policy,epoch in [('best_trained',best),('identity_allowed',original_epoch)]:
                    chosen=v[v.epoch.eq(epoch)].copy()
                    if len(chosen)!=42 or not chosen.model_id.str.startswith(arch+'-').all():raise ValueError('Wrong validation cases')
                    if not np.allclose(chosen.accuracy_gain,chosen.adapted_accuracy-chosen.baseline_accuracy):raise ValueError('Gain identity failed')
                    gain=100*float(chosen.accuracy_gain.mean())
                    rows.append(dict(architecture=arch,input=INPUTS[arch],seed=seed,method=method,policy=policy,
                        epoch=epoch,gain_pp=gain,frozen_accuracy_pct=100*float(chosen.baseline_accuracy.mean()),
                        adapted_accuracy_pct=100*float(chosen.adapted_accuracy.mean()),
                        helped=int((chosen.accuracy_gain>1e-10).sum()),harmed=int((chosen.accuracy_gain < -1e-10).sum())))
                    selected_cases.extend(dict(architecture=arch,input=INPUTS[arch],seed=seed,method=method,policy=policy,
                        model_id=x['model_id'],environment_id=x['environment_id'],gain_pp=100*x['accuracy_gain']) for x in chosen.to_dict('records'))
                    values[method,policy]=chosen.set_index(['model_id','environment_id']).baseline_accuracy.sort_index()
                h['architecture']=arch;h['input']=INPUTS[arch];h['seed']=seed;h['method']=method;curves.extend(h.to_dict('records'))
            if not values['Head','best_trained'].index.equals(values['LoRA','best_trained'].index):
                raise ValueError('Head and LoRA evaluated different source/graph cases')
            if not np.allclose(values['Head','best_trained'],values['LoRA','best_trained'],atol=1e-7,rtol=0):
                raise ValueError('Head/LoRA frozen baselines differ on their matched validation cases')
    scores=pd.DataFrame(rows);history=pd.DataFrame(curves);cases=pd.DataFrame(selected_cases)
    scores.to_csv(out/'per_seed_results.csv',index=False);cases.to_csv(out/'per_case_results.csv',index=False)
    history.to_csv(out/'training_curves.csv',index=False)
    for policy in ['best_trained','identity_allowed']:
        table=[]
        for arch in ARCHES:
            d=scores[(scores.architecture==arch)&scores.policy.eq(policy)]
            p=d.pivot(index='seed',columns='method',values='gain_pp').loc[SEEDS]
            diff=p.Head-p.LoRA
            table.append({'Backbone':NAMES[arch],'Input':INPUTS[arch],
                'Frozen validation accuracy (%)':f'{d.frozen_accuracy_pct.mean():.3f}',
                'Head gain (pp)':f'{p.Head.mean():+.3f} ± {p.Head.std(ddof=1):.3f}',
                'LoRA gain (pp)':f'{p.LoRA.mean():+.3f} ± {p.LoRA.std(ddof=1):.3f}',
                'Paired head minus LoRA (pp)':f'{diff.mean():+.3f} ± {diff.std(ddof=1):.3f}'})
        t=pd.DataFrame(table);t.to_csv(out/f'{policy}_table.csv',index=False)
        lines=['| '+' | '.join(t.columns)+' |','| '+' | '.join(['---']*len(t.columns))+' |']
        lines+=['| '+' | '.join(map(str,r))+' |' for r in t.to_numpy()]
        (out/f'{policy}_table.md').write_text('\n'.join(lines)+'\n')
    # Main plot shows both policies; paired seeds shown as dots, SD across seeds.
    fig,axes=plt.subplots(1,2,figsize=(12,4),sharey=True)
    for ax,policy,title in zip(axes,['best_trained','identity_allowed'],['Best trained checkpoint','Identity fallback allowed']):
        for j,method in enumerate(['Head','LoRA']):
            means=[];sds=[]
            for i,arch in enumerate(ARCHES):
                q=scores[(scores.architecture==arch)&scores.policy.eq(policy)&scores.method.eq(method)].set_index('seed').loc[SEEDS].gain_pp
                means.append(q.mean());sds.append(q.std(ddof=1))
                ax.scatter(np.full(3,i+(j-.5)*.3)+np.array([-.04,0,.04]),q,color='black',s=12,zorder=4,alpha=.65)
            ax.bar(np.arange(3)+(j-.5)*.3,means,.3,yerr=sds,capsize=3,label=method,color=['#2878B5','#E18B26'][j])
        ax.axhline(0,color='black',lw=1);ax.set_xticks(range(3),[NAMES[a] for a in ARCHES]);ax.set_title(title);ax.legend();ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Validation gain (percentage points)');fig.tight_layout();fig.savefig(out/'head_vs_lora_gain.png',dpi=200);plt.close(fig)
    # Full individual curves: never average over a changing seed population.
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    colors={'Head':'#2878B5','LoRA':'#E18B26'};styles=['-','--',':']
    for ax,arch in zip(axes,ARCHES):
        for method in ['Head','LoRA']:
            for seed,style in zip(SEEDS,styles):
                h=history[(history.architecture==arch)&history.method.eq(method)&history.seed.eq(seed)]
                ax.plot(h.epoch,h.train_loss,color=colors[method],ls=style,label=f'{method} / {seed}',lw=1.2)
        ax.set_title(NAMES[arch]+' / '+INPUTS[arch]);ax.set_xlabel('Epoch');ax.set_ylabel('Total training loss');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8,bbox_to_anchor=(1.02,1),loc='upper left');fig.tight_layout();fig.savefig(out/'training_loss.png',dpi=200);plt.close(fig)
    (out/'README.md').write_text('''# Experiment 2 Head versus LoRA
The main best_trained table chooses each method's highest mean trained validation gain, earliest exact tie, even if negative. The identity_allowed table uses the original saved selection, including epoch zero. Both policies are applied symmetrically to head and LoRA. Original training artifacts were not changed.

Inputs are fixed from validation: GCN C, GraphSAGE C, GAT A. Every method uses the same generator seeds 202801, 202802, 202803 and the same 42 validation model/graph cases per seed. Mean and sample SD are across the three seed-level means. The paired difference is Head minus LoRA for each matching seed; positive favours Head. The scatter dots show individual seeds.

Head is the whole final GNN layer, including attention parameters for GAT. LoRA is rank 4 at both layers. This compares both update location and parameterization, not location alone. Both use KL regularization, RMS scaling, cosine learning rate, maximum 100 epochs, patience 10.

The loss plot shows full individual-seed curves; different methods may stop at different epochs. LoRA references are checked against hashes saved during head training. No weights are loaded, no training or inference is rerun, and no test labels are scored. These are validation/development comparisons.
''')
    with zipfile.ZipFile(out/'experiment2_simple_report.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in out.iterdir():
            if p.is_file() and p.suffix!='.zip':z.write(p,p.name)
    return {'directory':str(out),'zip_bytes':(out/'experiment2_simple_report.zip').stat().st_size}


@app.function(image=image,cpu=2,memory=4096,timeout=1800,volumes={'/results':volume},retries=0)
def report(run_id:str,report_id:str):
    for value in [run_id,report_id]:
        if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in value):raise ValueError('Invalid run/report ID')
    volume.reload()
    result=analyze('/results',Path('/results')/run_id,Path('/results/reports')/report_id)
    volume.commit();return result


@app.local_entrypoint()
def main(run_id:str='arxiv-experiment2-head-001',report_id:str='arxiv-experiment2-report-001'):
    print(report.remote(run_id,report_id))
