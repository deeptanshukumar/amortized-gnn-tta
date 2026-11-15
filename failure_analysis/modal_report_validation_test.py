"""Volume-side KL validation/test reporting. No training or bulk checkpoint downloads."""
import os
from pathlib import Path
import modal
import modal_kl_backbones_parallel as training

HERE=Path(__file__).resolve().parent
app=modal.App('afml-kl-validation-test-report')
REPO=Path(os.environ.get('AFML_REPO',str(HERE.parent)))
image=(modal.Image.debian_slim(python_version='3.13')
    .pip_install('torch==2.11.0','torch-geometric==2.8.1','numpy==2.1.3',
                 'ogb==1.3.6','pandas==2.3.3','psutil==5.9.5','matplotlib==3.10.8')
    .env({'PYTHONPATH':'/repo/src:/repo/examples:/experiment','OMP_NUM_THREADS':'1',
          'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','MPLBACKEND':'Agg'})
    .add_local_dir(str(REPO/'src'),'/repo/src')
    .add_local_dir(str(REPO/'examples'),'/repo/examples')
    .add_local_file(str(REPO/'reports/real-data-2026-11-08/design/CODE.json'),'/experiment/original_CODE.json')
    .add_local_file(str(HERE/'worker.py'),'/experiment/worker.py')
    .add_local_file(str(HERE/'modal_input_comparison_l40s.py'),'/experiment/modal_input_comparison_l40s.py')
    .add_local_file(str(HERE/'modal_loss_comparison_l40s.py'),'/experiment/modal_loss_comparison_l40s.py')
    .add_local_file(str(HERE/'modal_kl_backbones_parallel.py'),'/experiment/modal_kl_backbones_parallel.py')
    .add_local_file(str(HERE/'modal_report_validation_test.py'),'/experiment/modal_report_validation_test.py'))
SEEDS=[202801,202802,202803]
ARCHES=['gcn','graphsage','gat']


def sha(path):
    import hashlib
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def choose_epoch(history,result):
    """Never select from test data. Exact identity selection, not rounded display."""
    if history.empty or history.epoch.le(0).any():raise ValueError('Missing trained epochs')
    if not history[['epoch','val_gain']].notna().all().all():raise ValueError('Invalid training history')
    original=int(result['selected_epoch'])
    if original>0:
        if original not in set(history.epoch):raise ValueError('Original selected epoch missing')
        return original,False
    # All trained candidates may be negative. Highest gain, earliest exact tie.
    best=history.sort_values(['val_gain','epoch'],ascending=[False,True]).iloc[0]
    return int(best.epoch),True


def locate_fit(root,parallel_run,gcn_first_run,arch,seed,c):
    if arch=='gcn' and seed==202801:return root/gcn_first_run/'kl'/f'seed_{seed}'/f'{c}-lora'
    return root/parallel_run/arch/f'seed_{seed}'/f'{c}-lora'


def freeze_plan(root,parallel_run,gcn_first_run,report_id,bank_path="/inputs/bank"):
    import json
    import pandas as pd
    from worker import write_json
    root=Path(root);out=root/'reports'/report_id;out.mkdir(parents=True,exist_ok=False)
    bank=Path(bank_path)
    manifest=json.loads((bank/'environment_manifest.json').read_text())
    index=json.loads((bank/'bank.json').read_text())
    if manifest.get('protocol')!='arxiv-transductive-v1' or manifest.get('dataset')!='ogbn-arxiv' or index['plan'].get('protocol')!='arxiv-formal-bank-v1':
        raise ValueError('Controlled-Arxiv formal bank required')
    fits=[];coefficients=set()
    bank_rows={r['id']:r for r in index['models']}
    for arch in ARCHES:
        for seed in SEEDS:
            for c in 'ABCD':
                folder=locate_fit(root,parallel_run,gcn_first_run,arch,seed,c)
                result=json.loads((folder/'RESULT.json').read_text())
                pins=json.loads((folder.parent/'study_context/SOURCE_PINS.json').read_text())
                if not pins or any(bank_rows.get(mid)!=row for mid,row in pins.items()):
                    raise ValueError('Training source checkpoint pins differ from this evaluation bank')
                if result.get('loss_kind')!='kl' or int(result['seed'])!=seed or result['conditioning']!=c:
                    raise ValueError(f'Wrong loss/seed/input: {folder}')
                if result.get('architecture',arch)!=arch:raise ValueError('Wrong backbone')
                coefficients.add(float(result['loss_coefficient']))
                history=pd.read_csv(folder/'history.csv');epoch,fallback=choose_epoch(history,result)
                cases=pd.read_csv(folder/'validation_cases.csv');chosen=cases[cases.epoch.eq(epoch)]
                if len(chosen)!=42 or chosen.duplicated(['model_id','environment_id']).any():raise ValueError('Invalid selected validation cases')
                if not chosen.model_id.str.startswith(arch+'-').all():raise ValueError('Wrong validation model IDs')
                epoch_file=folder/f'epoch_{epoch:03d}.pt'
                files={n:sha(folder/n) for n in ['best_generator.pt','scaling.pt','history.csv','validation_cases.csv']}
                files[epoch_file.name]=sha(epoch_file)
                fits.append(dict(architecture=arch,seed=seed,input=c,folder=str(folder),epoch=epoch,
                    original_epoch=int(result['selected_epoch']),identity_replaced=fallback,
                    validation_gain_pp=100*float(chosen.accuracy_gain.mean()),files_sha256=files))
    if len(coefficients)!=1:raise ValueError('KL coefficients differ between runs')
    # This is committed before any test scoring. No best-input selection on test.
    plan=dict(protocol='kl-all-inputs-validation-frozen-test-v1',report_id=report_id,
        parallel_run=parallel_run,gcn_first_run=gcn_first_run,seeds=SEEDS,architectures=ARCHES,
        fits=fits,KL_coefficient=coefficients.pop(),bank_sha256=sha(bank/'bank.json'),
        manifest_sha256=sha(bank/'environment_manifest.json'),
        selection='Original nonzero selected epoch; otherwise highest mean trained validation gain, earliest exact tie',
        validation='6 validation source models x 7 original validation environments; 2018 labels',
        test='6 held-out source models x 27 existing arxiv_generalization_manifest environments; official 2019+ labels',
        IM=dict(steps=30,lr=.01,rank=4,initialization_seeds=SEEDS,objective='information_maximization'),
        note='IM uses 3 genuine LoRA initializations; this is a reevaluation, not reuse of the old seed-801 score',
        aggregation='Average fixed model/environment cases within each seed, then mean and sample SD across 3 seeds',
        test_labels_for_selection=False)
    write_json(out/'FROZEN_EVALUATION_PLAN.json',plan)
    return plan


@app.function(image=image,cpu=2,memory=8192,timeout=1800,
              volumes={'/inputs':training.losses.base.inputs,'/results':training.losses.base.outputs},retries=0)
def prepare(parallel_run:str,gcn_first_run:str,report_id:str):
    from worker import write_json
    for name in [parallel_run,gcn_first_run,report_id]:
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in name):raise ValueError('Invalid run/report ID')
    plan=freeze_plan('/results',parallel_run,gcn_first_run,report_id)
    training.losses.base.outputs.commit()
    return plan['report_id']


@app.function(image=image,gpu='L40S',cpu=4,memory=32768,timeout=24*60*60,max_containers=3,
              volumes={'/inputs':training.losses.base.inputs,'/results':training.losses.base.outputs},retries=0)
def evaluate_arch(report_id:str,arch:str):
    import sys,json,time
    import numpy as np
    import pandas as pd
    import torch
    from types import SimpleNamespace
    from worker import prepare_runtime_source,write_json,write_csv,load_for_analysis
    # Separate worker snapshots must see the committed preflight plan.
    training.losses.base.outputs.reload()
    out=Path('/results/reports')/report_id
    plan=json.loads((out/'FROZEN_EVALUATION_PLAN.json').read_text());bank=Path('/inputs/bank')
    if sha(bank/'bank.json')!=plan['bank_sha256'] or sha(bank/'environment_manifest.json')!=plan['manifest_sha256']:raise ValueError('Source bank changed')
    repo=prepare_runtime_source('/repo','/tmp/afml-report-source')
    sys.path[:0]=[str(repo/'src'),str(repo/'examples')]
    from train_adapter import code_provenance
    expected=json.loads(Path('/experiment/original_CODE.json').read_text())['code']['source_hashes']
    if code_provenance()['source_hashes']!=expected:raise ValueError('Original source changed')
    from run_arxiv_experiments import Study,lora_config
    from amortized_gnn_tta.data.generalization import arxiv_generalization_manifest
    from amortized_gnn_tta.data.real_environments import ArxivEnvironments
    from amortized_gnn_tta.adaptation.benchmark import frozen_predictions,generated_predictions,iterative_predictions,execution_hardware
    from amortized_gnn_tta.adaptation_artifacts import state_sha256
    from amortized_gnn_tta.evaluation import score_logits
    from amortized_gnn_tta.training import software_versions
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('highest')
    dest=out/arch;dest.mkdir(exist_ok=False)
    study=Study(SimpleNamespace(bank=bank,data_root=Path('/inputs/arxiv'),output_dir=dest/'study_context',stop_after=2))
    if int(study.plan['teacher_steps'])!=30 or float(study.plan['teacher_lr'])!=.01 or study.plan['lora_rank']!=4:raise ValueError('Baseline recipe differs')
    fits=[f for f in plan['fits'] if f['architecture']==arch]
    for fit in fits:
        for name,digest in fit['files_sha256'].items():
            if sha(Path(fit['folder'])/name)!=digest:raise ValueError('Training artifact changed after selection freeze')
    rows=[];history_rows=[]
    # Generator validation metrics already exist for every rejected epoch.
    for fit in fits:
        folder=Path(fit['folder']);v=pd.read_csv(folder/'validation_cases.csv')
        for r in v[v.epoch.eq(fit['epoch'])].to_dict('records'):
            rows.append(dict(split='validation',architecture=arch,seed=fit['seed'],method=fit['input'],
                model_id=r['model_id'],environment_id=r['environment_id'],family=r['family'],severity=r['severity'],
                group='validation',baseline_accuracy=r['baseline_accuracy'],accuracy=r['adapted_accuracy'],gain=r['accuracy_gain']))
        h=pd.read_csv(folder/'history.csv');h['architecture']=arch;h['seed']=fit['seed'];h['input']=fit['input']
        history_rows.extend(h.to_dict('records'))
    write_csv(dest/'training_history.csv',history_rows)
    write_json(dest/'RUNTIME.json',dict(hardware=execution_hardware('cuda'),software=software_versions(),plan_sha256=sha(out/'FROZEN_EVALUATION_PLAN.json')))
    test_manifest=arxiv_generalization_manifest(study.manifest)
    test_loader=ArxivEnvironments(study.loader.dataset,test_manifest,allow_final=True)
    if len(test_manifest['environments'])!=27:raise ValueError('Unexpected predefined test protocol')
    write_json(dest/'TEST_ENVIRONMENTS.json',test_manifest)
    def record(split,seed,method,mid,env,base,acc):
        rows.append(dict(split=split,architecture=arch,seed=seed,method=method,model_id=mid,
            environment_id=env['id'],family=env['family'],severity=env['severity'],
            group=env.get('generalization_group','validation'),baseline_accuracy=base,accuracy=acc,gain=acc-base))
    for split,role,loader,manifest,envs in [
        ('validation','validation',study.loader,study.manifest,study.validation),
        ('test','test',test_loader,test_manifest,test_manifest['environments'])]:
        models=study.sources(arch,role)
        if len(models)!=6:raise ValueError('Expected six source instances per evaluation role')
        # Rebuild generators from each saved recipe; replace only epoch state.
        generators={}
        if split=='test':
            example=next(iter(models.values()))
            for fit in fits:
                g,d,p=load_for_analysis(fit['folder'],example,device='cuda')
                checkpoint=torch.load(Path(fit['folder'])/f"epoch_{fit['epoch']:03d}.pt",map_location='cpu',weights_only=True)
                if checkpoint['epoch']!=fit['epoch']:raise ValueError('Wrong epoch checkpoint')
                g.load_state_dict(checkpoint['state_dict'],strict=True);g.eval()
                generators[fit['seed'],fit['input']]=(g,d,p['target_scope'])
        for env in envs:
            bundle=loader(manifest,env['id']);graph=bundle.unlabeled_input(env['id']);indices=bundle.select(env['id'],'eval')
            for mid,source in models.items():
                source_hash=state_sha256(source.state_dict())
                frozen=frozen_predictions(source,graph,device='cuda')
                # No labels are passed to either generator or iterative adaptation.
                frozen_classes=frozen.logits[indices.to(frozen.logits.device)].argmax(-1).cpu();del frozen
                predictions={}
                for seed in SEEDS:
                    torch.manual_seed(seed)
                    im=iterative_predictions(source,graph,indices=indices,steps=30,lr=.01,
                        peft=lora_config(study.plan),objective='information_maximization',initialization_seed=seed,device='cuda')
                    predictions[seed,'IM-LoRA']=im.logits[indices.to(im.logits.device)].argmax(-1).cpu();del im
                if split=='test':
                    for (seed,c),(g,d,scope) in generators.items():
                        pred=generated_predictions(source,graph,d,g,scope=scope,device='cuda')
                        predictions[seed,c]=pred.logits[indices.to(pred.logits.device)].argmax(-1).cpu();del pred
                # Scoring is after ALL predictions for this source/graph case.
                labels=bundle.evaluation_labels(env['id'],'eval').cpu()
                base=float(frozen_classes.eq(labels).float().mean())
                for seed in SEEDS:record(split,seed,'Frozen',mid,env,base,base)
                for (seed,method),classes in predictions.items():
                    record(split,seed,method,mid,env,base,float(classes.eq(labels).float().mean()))
                if state_sha256(source.state_dict())!=source_hash:raise ValueError('Evaluation mutated source weights')
                write_csv(dest/'case_metrics.csv',rows)
                write_json(dest/'STATUS.json',dict(split=split,model=mid,environment=env['id'],cases_saved=len(rows)))
                training.losses.base.outputs.commit()
                print(f'[{arch}] {split} {env["id"]} {mid} complete',flush=True)
        del generators
    write_json(dest/'COMPLETE.json',dict(case_rows=len(rows),test_scored=True,selection_used_test_labels=False))
    training.losses.base.outputs.commit()
    return arch


def make_summary(cases):
    """One number per generator/IM seed, then sample SD across seeds."""
    import pandas as pd
    seed=cases.groupby(['split','architecture','method','seed'],as_index=False).agg(
        accuracy=('accuracy','mean'),gain=('gain','mean'),frozen=('baseline_accuracy','mean'),cases=('gain','size'))
    counts=seed.groupby(['split','architecture','method']).seed.nunique()
    if not counts.eq(3).all():raise ValueError('Not all methods have exactly three seeds')
    stats=seed.groupby(['split','architecture','method'],as_index=False).agg(
        accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),
        gain_mean=('gain','mean'),gain_sd=('gain','std'),frozen_mean=('frozen','mean'))
    return seed,stats


@app.function(image=image,cpu=2,memory=8192,timeout=1800,
              volumes={'/results':training.losses.base.outputs},retries=0)
def summarize(report_id:str):
    training.losses.base.outputs.reload()
    return render_report(Path('/results/reports')/report_id,training.losses.base.outputs.commit)


def render_report(root,commit=lambda:None):
    import json,zipfile
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from worker import write_json
    root=Path(root)
    plan=json.loads((root/'FROZEN_EVALUATION_PLAN.json').read_text())
    for arch in ARCHES:
        if not (root/arch/'COMPLETE.json').exists():raise ValueError('Incomplete architecture evaluation')
    cases=pd.concat([pd.read_csv(root/a/'case_metrics.csv') for a in ARCHES],ignore_index=True)
    histories=pd.concat([pd.read_csv(root/a/'training_history.csv') for a in ARCHES],ignore_index=True)
    seed,stats=make_summary(cases)
    cases.to_csv(root/'per_case_metrics.csv',index=False);seed.to_csv(root/'per_seed_metrics.csv',index=False)
    stats.to_csv(root/'summary_numeric.csv',index=False);histories.to_csv(root/'training_curves.csv',index=False)
    methods=['IM-LoRA','A','B','C','D'];names={'gcn':'GCN','graphsage':'GraphSAGE','gat':'GAT'}
    for split in ['validation','test']:
        table=[]
        for arch in ARCHES:
            s=stats[(stats.split==split)&(stats.architecture==arch)].set_index('method')
            row={'Backbone':names[arch],'Frozen accuracy (%)':f"{100*s.loc['Frozen','accuracy_mean']:.3f}"}
            for method in methods:row[method+' gain (pp)']=f"{100*s.loc[method,'gain_mean']:+.3f} ± {100*s.loc[method,'gain_sd']:.3f}"
            table.append(row)
        frame=pd.DataFrame(table);frame.to_csv(root/f'{split}_table.csv',index=False)
        # Markdown without optional tabulate dependency.
        lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |']
        lines+=['| '+' | '.join(map(str,row))+' |' for row in frame.to_numpy()]
        (root/f'{split}_table.md').write_text('\n'.join(lines)+'\n')
        fig,ax=plt.subplots(figsize=(12,4.5));x=np.arange(3);width=.15
        for j,method in enumerate(methods):
            values=[];errors=[]
            for arch in ARCHES:
                q=stats[(stats.split==split)&(stats.architecture==arch)&(stats.method==method)].iloc[0]
                values.append(100*q.gain_mean);errors.append(100*q.gain_sd)
            ax.bar(x+(j-2)*width,values,width,yerr=errors,label=method,capsize=3)
        ax.axhline(0,color='black',lw=1);ax.set_xticks(x,[names[a] for a in ARCHES]);ax.set_ylabel('Accuracy gain (percentage points)')
        ax.set_title(f'{split.title()} gain — mean ± sample SD across three seeds');ax.legend(ncol=5);ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(root/f'{split}_gain.png',dpi=180);plt.close(fig)
    # Small gains must remain visible if the IM baseline is strongly negative.
    fig,axes=plt.subplots(1,2,figsize=(13,4))
    for ax,split in zip(axes,['validation','test']):
        for j,c in enumerate('ABCD'):
            q=stats[(stats.split==split)&(stats.method==c)].set_index('architecture').loc[ARCHES]
            ax.bar(np.arange(3)+(j-1.5)*.19,100*q.gain_mean,.19,yerr=100*q.gain_sd,label=c,capsize=3)
        ax.set_xticks(range(3),[names[a] for a in ARCHES]);ax.axhline(0,color='black',lw=1);ax.set_title(split.title());ax.set_ylabel('Generator gain (pp)');ax.legend()
    fig.tight_layout();fig.savefig(root/'generator_gain_detail.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(3,4,figsize=(16,10))
    for i,arch in enumerate(ARCHES):
        for j,c in enumerate('ABCD'):
            ax=axes[i,j]
            for s in SEEDS:
                h=histories[(histories.architecture==arch)&(histories.input==c)&(histories.seed==s)]
                ax.plot(h.epoch,h.train_loss,label=str(s));f=next(x for x in plan['fits'] if x['architecture']==arch and x['seed']==s and x['input']==c)
                ax.axvline(f['epoch'],ls=':',lw=.7,alpha=.5)
            ax.set_title(names[arch]+' / '+c);ax.set_xlabel('Epoch');ax.set_ylabel('Total training loss');ax.grid(alpha=.2)
    axes[0,0].legend(title='Seed');fig.tight_layout();fig.savefig(root/'training_loss.png',dpi=160);plt.close(fig)
    (root/'README.md').write_text('''# KL validation and test report
All reported generator choices were frozen using validation before test scoring. Each architecture/input uses seeds 202801, 202802 and 202803. Mean and sample SD are computed across seed-level means over fixed model/environment cases. Reused graph/model cases are not independent datasets.

IMPORTANT: If the original selected epoch was zero, this report uses the best TRAINED epoch, even if harmful. This is a forced-adaptation report, not the original do-no-harm policy. FROZEN_EVALUATION_PLAN.json lists both selections. Tiny positive gains rounded to 0.00 are not treated as identity selections.

Validation: six validation models, seven original conditions, 2018 labels. Test: six separate test models, 27 predefined generalization conditions (fresh draws, stronger shifts and new operators), official 2019+ labels. Their averages therefore describe different conditions. Test-group metrics are supplied separately. No test-driven checkpoint or input selection.

Frozen accuracy is a fixed-case mean. Each A–D and IM gain is measured against frozen on its own case, then aggregated. IM-LoRA uses 30 steps, lr .01, rank 4 at both layers, and three LoRA initialization seeds 202801–202803. This differs from the historical seed-801-only baseline. No target labels enter iterative optimization or generator inputs. Generators had offline labeled practice access; IM did not.

Plots: validation_gain.png and test_gain.png include IM; generator_gain_detail.png makes tiny generator gains visible; training_loss.png shows each seed's TOTAL training loss, with selected epochs marked. training_curves.csv also contains CE, frozen CE, KL and learning-rate columns where recorded. No new training was performed.
''')
    groups=cases[cases.split.eq('test')].groupby(['architecture','method','seed','group'],as_index=False).agg(gain=('gain','mean'),accuracy=('accuracy','mean'))
    groups.to_csv(root/'test_groups_by_seed.csv',index=False)
    bundle_files=[p for p in root.iterdir() if p.is_file() and p.suffix in {'.png','.csv','.md','.json'}]
    with zipfile.ZipFile(root/'small_report.zip','w',compression=zipfile.ZIP_DEFLATED,strict_timestamps=False) as z:
        for p in bundle_files:z.write(p,p.name)
    write_json(root/'REPORT_COMPLETE.json',dict(test_scored=True,files=[p.name for p in bundle_files],archive='small_report.zip'))
    commit()
    return {'report':str(root),'zip_bytes':(root/'small_report.zip').stat().st_size}


@app.function(image=image,cpu=1,memory=2048,timeout=24*60*60,retries=0)
def orchestrate(parallel_run:str,gcn_first_run:str,report_id:str):
    prepare.remote(parallel_run,gcn_first_run,report_id)
    calls=[evaluate_arch.spawn(report_id,arch) for arch in ARCHES]
    errors=[]
    for arch,call in zip(ARCHES,calls):
        try:print(arch,call.get())
        except Exception as error:errors.append((arch,str(error)))
    if errors:raise RuntimeError(f'Partial evaluation preserved; no summary of incomplete results: {errors}')
    print(summarize.remote(report_id))


@app.local_entrypoint()
def main(report_id:str='',parallel_run:str='arxiv-kl-backbones-001',gcn_first_run:str='arxiv-new-losses-001'):
    if not report_id:raise ValueError('Supply a unique --report-id')
    print(orchestrate.remote(parallel_run,gcn_first_run,report_id))
