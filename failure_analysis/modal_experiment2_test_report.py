"""Compact Experiment 2 validation/test report. Reuses frozen LoRA test scores."""
from pathlib import Path
import os,modal
import modal_report_validation_test as previous
app=modal.App('afml-experiment2-validation-test')
volume=previous.training.losses.base.outputs
HERE=Path(__file__).resolve().parent
image=previous.image.add_local_file(str(HERE/'modal_experiment2_test_report.py'),'/experiment/modal_experiment2_test_report.py')
ARCHES=['gcn','graphsage','gat'];INPUTS={'gcn':'C','graphsage':'C','gat':'A'}
NAMES={'gcn':'GCN','graphsage':'GraphSAGE','gat':'GAT'};SEEDS=[202801,202802,202803]


def select_epochs(history,result):
    # Same validation-only rule as the preceding Experiment 2 report.
    best=int(history.sort_values(['val_gain','epoch'],ascending=[False,True]).iloc[0].epoch)
    return {'best_trained':best,'identity_allowed':int(result['selected_epoch'])}


def freeze(root,head_run,reference_report,report_id,bank_path='/inputs/bank'):
    import json,pandas as pd
    from worker import write_json
    root=Path(root);bank=Path(bank_path)
    old=root/'reports'/reference_report
    oldplan=json.loads((old/'FROZEN_EVALUATION_PLAN.json').read_text())
    if previous.sha(bank/'bank.json')!=oldplan['bank_sha256'] or previous.sha(bank/'environment_manifest.json')!=oldplan['manifest_sha256']:
        raise ValueError('Current bank differs from prior LoRA test evaluation')
    cache=pd.read_csv(old/'per_case_metrics.csv')
    fits=[];val=[];curves=[];lora_selections={a:{} for a in ARCHES}
    for arch in ARCHES:
        source=root/head_run/arch
        if not (source/'COMPLETE.json').exists():raise ValueError('Head campaign incomplete')
        setup=json.loads((source/'EXPERIMENT_PLAN.json').read_text())
        if setup['input']!=INPUTS[arch] or setup['seeds']!=SEEDS:raise ValueError('Wrong input/seeds')
        for seed in SEEDS:
            ref=setup['references'][str(seed)]
            lora=root/Path(ref['folder']).relative_to('/results')
            if previous.sha(lora/'RESULT.json')!=ref['result_sha256'] or previous.sha(lora/'history.csv')!=ref['history_sha256']:raise ValueError('LoRA reference changed')
            head=source/f'seed_{seed}'/f'{INPUTS[arch]}-head'
            for method,folder in [('Head',head),('LoRA',lora)]:
                result=json.loads((folder/'RESULT.json').read_text());h=pd.read_csv(folder/'history.csv');v=pd.read_csv(folder/'validation_cases.csv')
                if result['seed']!=seed or result['conditioning']!=INPUTS[arch] or result['loss_kind']!='kl' or result['update']!=('head' if method=='Head' else 'lora'):
                    raise ValueError('Wrong training artifacts')
                epochs=select_epochs(h,result)
                h['architecture']=arch;h['seed']=seed;h['method']=method;curves.extend(h.to_dict('records'))
                for policy,epoch in epochs.items():
                    chosen=v[v.epoch.eq(epoch)]
                    if len(chosen)!=42 or chosen.duplicated(['model_id','environment_id']).any():raise ValueError('Invalid validation cases')
                    val.extend(dict(split='validation',architecture=arch,seed=seed,method=method,policy=policy,
                        model_id=r['model_id'],environment_id=r['environment_id'],baseline_accuracy=r['baseline_accuracy'],accuracy=r['adapted_accuracy'],gain=r['accuracy_gain']) for r in chosen.to_dict('records'))
                if method=='Head':
                    files={n:previous.sha(folder/n) for n in ['best_generator.pt','scaling.pt','history.csv','validation_cases.csv']}
                    for epoch in set(epochs.values()):files[f'epoch_{epoch:03d}.pt']=previous.sha(folder/f'epoch_{epoch:03d}.pt')
                    fits.append(dict(architecture=arch,seed=seed,input=INPUTS[arch],folder=str(folder),epochs=epochs,files_sha256=files))
                else:
                    lora_selections[arch][str(seed)]=epochs
                    saved=next(f for f in oldplan['fits'] if f['architecture']==arch and f['seed']==seed and f['input']==INPUTS[arch])
                    for name,digest in saved['files_sha256'].items():
                        if previous.sha(folder/name)!=digest:raise ValueError('Previously evaluated LoRA artifact changed')
                    if epochs['best_trained']!=saved['epoch']:raise ValueError('LoRA best-trained epoch was not evaluated in prior test report')
                    if epochs['identity_allowed'] not in [0,saved['epoch']]:raise ValueError('LoRA selected epoch unavailable in test cache')
    # Require a complete cached test grid before using it.
    for arch in ARCHES:
        for seed in SEEDS:
            for method in ['Frozen',INPUTS[arch]]:
                q=cache[(cache.split=='test')&(cache.architecture==arch)&(cache.seed==seed)&(cache.method==method)]
                if len(q)!=162 or q.duplicated(['model_id','environment_id']).any():raise ValueError('Incomplete LoRA test cache')
    out=root/'reports'/report_id;out.mkdir(parents=True,exist_ok=False)
    plan=dict(experiment=2,head_run=head_run,reference_report=reference_report,seeds=SEEDS,inputs=INPUTS,
        fits=fits,lora_selections=lora_selections,bank_sha256=oldplan['bank_sha256'],manifest_sha256=oldplan['manifest_sha256'],
        prior_test_cache_sha256=previous.sha(old/'per_case_metrics.csv'),
        note='Epochs selected using validation only. Test scores are descriptive follow-up evaluations, not untouched after Experiment 1.',
        test_protocol='existing 27-condition arxiv_generalization_manifest; six test GNNs; 2019+ labels',
        selection='best_trained forces trained epochs including negative; identity_allowed preserves original selection')
    write_json(out/'FROZEN_HEAD_SELECTION.json',plan)
    pd.DataFrame(val).to_csv(out/'validation_cases.csv',index=False);pd.DataFrame(curves).to_csv(out/'training_curves.csv',index=False)
    # Freeze a compact local copy of the already scored test cache for this report.
    cache[cache.split.eq('test')].to_csv(out/'prior_test_cache.csv',index=False)
    return plan


@app.function(image=image,cpu=2,memory=8192,timeout=1800,volumes={'/inputs':previous.training.losses.base.inputs,'/results':volume},retries=0)
def prepare(head_run:str,reference_report:str,report_id:str):
    for name in [head_run,reference_report,report_id]:
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):raise ValueError('Invalid ID')
    volume.reload();freeze('/results',head_run,reference_report,report_id);volume.commit()
    return report_id


@app.function(image=image,gpu='L40S',cpu=4,memory=32768,timeout=24*60*60,max_containers=3,
    volumes={'/inputs':previous.training.losses.base.inputs,'/results':volume},retries=0)
def evaluate_head(report_id:str,arch:str):
    import sys,json,numpy as np,pandas as pd,torch
    from types import SimpleNamespace
    from worker import prepare_runtime_source,write_json,load_for_analysis
    volume.reload();out=Path('/results/reports')/report_id
    plan=json.loads((out/'FROZEN_HEAD_SELECTION.json').read_text());bank=Path('/inputs/bank')
    if previous.sha(bank/'bank.json')!=plan['bank_sha256'] or previous.sha(bank/'environment_manifest.json')!=plan['manifest_sha256']:raise ValueError('Bank changed')
    repo=prepare_runtime_source('/repo','/tmp/afml-experiment2-test-source');sys.path[:0]=[str(repo/'src'),str(repo/'examples')]
    from train_adapter import code_provenance
    if code_provenance()['source_hashes']!=json.loads(Path('/experiment/original_CODE.json').read_text())['code']['source_hashes']:raise ValueError('Original source changed')
    from run_arxiv_experiments import Study
    from amortized_gnn_tta.data.generalization import arxiv_generalization_manifest
    from amortized_gnn_tta.data.real_environments import ArxivEnvironments
    from amortized_gnn_tta.adaptation.benchmark import frozen_predictions,generated_predictions
    from amortized_gnn_tta.adaptation_artifacts import state_sha256
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True;torch.set_float32_matmul_precision('highest')
    dest=out/arch;dest.mkdir(exist_ok=False)
    study=Study(SimpleNamespace(bank=bank,data_root=Path('/inputs/arxiv'),output_dir=dest/'study_context',stop_after=2))
    manifest=arxiv_generalization_manifest(study.manifest)
    if len(manifest['environments'])!=27:raise ValueError('Unexpected test protocol')
    loader=ArxivEnvironments(study.loader.dataset,manifest,allow_final=True)
    models=study.sources(arch,'test');example=next(iter(models.values()));generators={};selections={}
    for f in [f for f in plan['fits'] if f['architecture']==arch]:
        folder=Path(f['folder'])
        for name,digest in f['files_sha256'].items():
            if previous.sha(folder/name)!=digest:raise ValueError('Head checkpoint changed after freeze')
        selections[f['seed']]=f['epochs']
        for epoch in set(f['epochs'].values())-{0}:
            g,d,p=load_for_analysis(folder,example,device='cuda')
            state=torch.load(folder/f'epoch_{epoch:03d}.pt',map_location='cpu',weights_only=True)
            if state['epoch']!=epoch or p['target_scope']!='head':raise ValueError('Wrong head checkpoint')
            g.load_state_dict(state['state_dict'],strict=True);g.eval();generators[f['seed'],epoch]=(g,d)
    cached=pd.read_csv(out/'prior_test_cache.csv');cached=cached[cached.architecture.eq(arch)]
    rows=[];drift=[]
    for env in manifest['environments']:
        bundle=loader(manifest,env['id']);graph=bundle.unlabeled_input(env['id']);indices=bundle.select(env['id'],'eval')
        for mid,model in models.items():
            source_hash=state_sha256(model.state_dict())
            pred=frozen_predictions(model,graph,device='cuda');base_classes=pred.logits[indices.to(pred.logits.device)].argmax(-1).cpu();del pred
            predictions={}
            for key,(g,d) in generators.items():
                pred=generated_predictions(model,graph,d,g,scope='head',device='cuda')
                predictions[key]=pred.logits[indices.to(pred.logits.device)].argmax(-1).cpu();del pred
            # Test labels are read only for scoring, after all predictions.
            labels=bundle.evaluation_labels(env['id'],'eval').cpu();base=float(base_classes.eq(labels).float().mean())
            for seed in SEEDS:
                old=cached[(cached.seed==seed)&(cached.model_id==mid)&(cached.environment_id==env['id'])]
                frozen=old[old.method.eq('Frozen')].iloc[0];lora=old[old.method.eq(INPUTS[arch])].iloc[0]
                difference=base-float(frozen.accuracy)
                if abs(difference)>2/len(labels)+1e-8:raise ValueError('Unexpected frozen prediction drift relative to LoRA cache')
                drift.append(dict(seed=seed,model_id=mid,environment_id=env['id'],accuracy_difference=difference))
                for policy,epoch in selections[seed].items():
                    head_acc=base if epoch==0 else float(predictions[seed,epoch].eq(labels).float().mean())
                    # LoRA identity policy: GAT's original epoch-zero result stays identity.
                    lora_identity=(policy=='identity_allowed' and plan['lora_selections'][arch][str(seed)]['identity_allowed']==0)
                    for method,acc,baseline in [('Head',head_acc,base),('LoRA',float(frozen.accuracy) if lora_identity else float(lora.accuracy),float(frozen.accuracy))]:
                        rows.append(dict(split='test',architecture=arch,seed=seed,method=method,policy=policy,model_id=mid,
                            environment_id=env['id'],baseline_accuracy=baseline,accuracy=acc,gain=acc-baseline))
            if state_sha256(model.state_dict())!=source_hash:raise ValueError('Inference mutated source weights')
            pd.DataFrame(rows).to_csv(dest/'test_cases.csv',index=False)
            print(f'[E2 test {arch}] {env["id"]} {mid}',flush=True);volume.commit()
    pd.DataFrame(drift).to_csv(dest/'frozen_score_drift.csv',index=False)
    write_json(dest/'COMPLETE.json',dict(test_scored=True,case_rows=len(rows),max_frozen_drift=max(abs(x['accuracy_difference']) for x in drift)))
    volume.commit();return arch


def plot_loss(history,out):
    import numpy as np,matplotlib.pyplot as plt
    colors={'Head':'#2878B5','LoRA':'#E18B26'}
    for short in [False,True]:
        fig,axes=plt.subplots(1,3,figsize=(13,3.8))
        for ax,arch in zip(axes,ARCHES):
            for method in ['Head','LoRA']:
                h=history[(history.architecture==arch)&history.method.eq(method)]
                grid=h.pivot(index='epoch',columns='seed',values='train_loss').reindex(columns=SEEDS).dropna()
                if short:grid=grid[grid.index<=10]
                mean=grid.mean(axis=1);sd=grid.std(axis=1,ddof=1)
                ax.plot(grid.index,mean,label=method,color=colors[method],lw=1.8)
                ax.fill_between(grid.index,mean-sd,mean+sd,color=colors[method],alpha=.15)
            ax.set_title(NAMES[arch]+' / '+INPUTS[arch]);ax.set_xlabel('Epoch');ax.set_ylabel('Total training loss');ax.grid(alpha=.2);ax.legend(frameon=False)
        fig.tight_layout();fig.savefig(Path(out)/('training_loss_first10.png' if short else 'training_loss.png'),dpi=220);plt.close(fig)


def plot_gains(seed,out):
    import numpy as np,matplotlib.pyplot as plt
    for policy,name in [('best_trained','head_vs_lora_gain.png'),('identity_allowed','identity_allowed_gain.png')]:
        splits=[s for s in ['validation','test'] if s in set(seed.split)]
        fig,axes=plt.subplots(1,len(splits),figsize=(6*len(splits),3.8),squeeze=False,sharey=True)
        for ax,split in zip(axes[0],splits):
            for j,method in enumerate(['Head','LoRA']):
                values=[];errors=[]
                for arch in ARCHES:
                    p=seed[(seed.split==split)&(seed.architecture==arch)&seed.method.eq(method)&seed.policy.eq(policy)]
                    values.append(p.gain_pp.mean());errors.append(p.gain_pp.std(ddof=1))
                ax.bar(np.arange(3)+(j-.5)*.3,values,.3,yerr=errors,capsize=4,label=method,color=['#2878B5','#E18B26'][j])
            ax.axhline(0,color='black',lw=.9);ax.set_xticks(range(3),[NAMES[a] for a in ARCHES]);ax.set_title(split.title());ax.set_ylabel('Gain over frozen GNN (pp)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False)
        fig.tight_layout();fig.savefig(Path(out)/name,dpi=220);plt.close(fig)


def render(root):
    import json,zipfile,pandas as pd
    root=Path(root);v=pd.read_csv(root/'validation_cases.csv');t=pd.concat([pd.read_csv(root/a/'test_cases.csv') for a in ARCHES])
    cases=pd.concat([v,t],ignore_index=True);cases.to_csv(root/'per_case_results.csv',index=False)
    counts=cases.groupby(['split','architecture','seed','method','policy']).size()
    if not all(n==(42 if s=='validation' else 162) for (s,*_),n in counts.items()):raise ValueError('Incomplete scored cases')
    seed=cases.groupby(['split','architecture','seed','method','policy'],as_index=False).agg(gain_pp=('gain',lambda s:100*s.mean()),accuracy_pct=('accuracy',lambda s:100*s.mean()),frozen_pct=('baseline_accuracy',lambda s:100*s.mean()))
    seed.to_csv(root/'per_seed_results.csv',index=False)
    for split in ['validation','test']:
        for policy in ['best_trained','identity_allowed']:
            rows=[]
            for arch in ARCHES:
                p=seed[(seed.split==split)&(seed.architecture==arch)&seed.policy.eq(policy)]
                g=p.pivot(index='seed',columns='method',values='gain_pp').loc[SEEDS];delta=g.Head-g.LoRA
                rows.append({'Backbone':NAMES[arch],'Input':INPUTS[arch],'Frozen accuracy (%)':f'{p.frozen_pct.mean():.3f}',
                    'Head gain (pp)':f'{g.Head.mean():+.3f} ± {g.Head.std(ddof=1):.3f}',
                    'LoRA gain (pp)':f'{g.LoRA.mean():+.3f} ± {g.LoRA.std(ddof=1):.3f}',
                    'Paired head minus LoRA (pp)':f'{delta.mean():+.3f} ± {delta.std(ddof=1):.3f}'})
            table=pd.DataFrame(rows);base=f'{split}_{policy}_table';table.to_csv(root/(base+'.csv'),index=False)
            lines=['| '+' | '.join(table.columns)+' |','| '+' | '.join(['---']*len(table.columns))+' |']+['| '+' | '.join(map(str,row))+' |' for row in table.to_numpy()]
            (root/(base+'.md')).write_text('\n'.join(lines)+'\n')
    drift_files=[root/a/'frozen_score_drift.csv' for a in ARCHES]
    if all(p.exists() for p in drift_files):
        drift=pd.concat([pd.read_csv(p).assign(architecture=a) for a,p in zip(ARCHES,drift_files)],ignore_index=True)
        drift.to_csv(root/'frozen_score_drift.csv',index=False)
    h=pd.read_csv(root/'training_curves.csv');plot_loss(h,root);plot_gains(seed,root)
    (root/'README.md').write_text('''# Experiment 2 validation and test
Main tables/plot use best trained validation checkpoints, even if negative. Companion tables/plot retain original identity-allowed selection. These rules are symmetric for head and LoRA. All selections were frozen from validation before new head test scoring. This is a follow-up evaluation after Experiment 1 test results were inspected.

Test uses the same 27 conditions and six test GNNs as arxiv-kl-report-001 (162 cases per seed); validation uses seven conditions and six validation GNNs (42 cases). LoRA test metrics are reused only after exact epoch/artifact checks. Head-only inference is newly evaluated. No training, iterative baseline optimization or test-driven selection was performed.

Bars are mean ± sample SD across seeds 202801–202803; there are no seed dots. Loss curves show a mean line and SD band over epochs with ALL three seeds available for that method. They stop at the shortest seed trajectory, avoiding a changing seed population. training_loss_first10.png provides an aligned first-ten-epoch view. Full curves remain in CSV.

Head is the entire final GNN layer; LoRA is rank 4 at both layers. The experiment changes update location and parameterization. Frozen score drift against the previous cached evaluation is audited; a discrepancy beyond two node predictions per case fails evaluation.\n''')
    with zipfile.ZipFile(root/'experiment2_report.zip','w',zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
        for p in root.iterdir():
            if p.is_file() and p.suffix in {'.csv','.md','.png','.json'}:z.write(p,p.name)
    return {'report':str(root),'zip_bytes':(root/'experiment2_report.zip').stat().st_size}


@app.function(image=image,cpu=2,memory=8192,timeout=1800,volumes={'/results':volume},retries=0)
def summarize(report_id:str):
    volume.reload();out=Path('/results/reports')/report_id
    for arch in ARCHES:
        if not (out/arch/'COMPLETE.json').exists():raise ValueError('Test evaluation incomplete')
    result=render(out);volume.commit();return result


@app.function(image=image,cpu=1,memory=2048,timeout=24*60*60,retries=0)
def orchestrate(head_run:str,reference_report:str,report_id:str):
    prepare.remote(head_run,reference_report,report_id)
    calls={a:evaluate_head.spawn(report_id,a) for a in ARCHES};errors=[]
    for a,call in calls.items():
        try:print(a,call.get())
        except Exception as error:errors.append((a,str(error)))
    if errors:raise RuntimeError(f'Partial results preserved: {errors}')
    return summarize.remote(report_id)


@app.local_entrypoint()
def main(head_run:str='arxiv-experiment2-head-001',reference_report:str='arxiv-kl-report-001',report_id:str='arxiv-experiment2-test-report-001'):
    print(orchestrate.remote(head_run,reference_report,report_id))
