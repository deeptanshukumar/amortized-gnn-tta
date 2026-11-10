"""Experiment 2: head-only versus EXISTING LoRA, validation only, no new test scoring.

GCN=C, GraphSAGE=C, GAT=A. Three L40S workers; seeds sequential per backbone.
Head means the entire final GNN layer (including GAT attention parameters),
not necessarily a plain linear classifier. This compares site AND parameterization.
"""
import os
from pathlib import Path
import modal
import modal_kl_backbones_parallel as common
HERE=Path(__file__).resolve().parent
app=modal.App('afml-experiment2-head')
image=common.image.add_local_file(str(HERE/'modal_experiment2_head.py'),'/experiment/modal_experiment2_head.py')
INPUTS={'gcn':'C','graphsage':'C','gat':'A'}
SEEDS=[202801,202802,202803]


class PrefixLog:
    def __init__(self,stream,prefix):self.stream=stream;self.prefix=prefix;self.start=True
    def write(self,text):
        for part in text.splitlines(keepends=True):
            if self.start and part:self.stream.write(self.prefix)
            self.stream.write(part);self.start=part.endswith('\n')
        return len(text)
    def flush(self):self.stream.flush()


def fit_head(study,architecture,conditioning,seed,folder,commit,coefficient=1.,*,max_epochs=100,patience=10):
    """Reuse the KL backbone runner; override only the generator update site."""
    import json
    import torch
    from unittest.mock import patch
    import run_arxiv_experiments as runner
    from worker import write_json
    original_factory=runner.create_generator
    def head_factory(model,c,update,method,plan):
        if update!='lora' or method!='hypernetwork':raise ValueError('Unexpected base runner call')
        return original_factory(model,c,'head',method,plan)
    with patch.object(runner,'create_generator',head_factory):
        result=common.run_fit(study,architecture,conditioning,seed,folder,commit,coefficient,
                              max_epochs=max_epochs,patience=patience)
    folder=Path(folder)
    result.update(update='head',experiment=2)
    for filename in ['RECIPE.json','LOSS_RECIPE.json']:
        recipe=json.loads((folder/filename).read_text())
        recipe.update(update='head',experiment=2,comparison='whole final-layer additive update versus both-layer rank-4 LoRA')
        write_json(folder/filename,recipe)
    checkpoint=torch.load(folder/'best_generator.pt',map_location='cpu',weights_only=True)
    if checkpoint['target_scope']!='head' or 'lora_space' in checkpoint:
        raise ValueError('Expected an actual direct head generator, not LoRA')
    checkpoint['provenance'].update(update='head',experiment=2)
    torch.save(checkpoint,folder/'best_generator.tmp');os.replace(folder/'best_generator.tmp',folder/'best_generator.pt')
    write_json(folder/'RESULT.json',result);write_json(folder/'STATUS.json',dict(state='complete',**result));commit()
    return result


def reference_folder(root,parallel_run,gcn_first_run,arch,seed):
    c=INPUTS[arch]
    if arch=='gcn' and seed==202801:return Path(root)/gcn_first_run/'kl'/f'seed_{seed}'/f'{c}-lora'
    return Path(root)/parallel_run/arch/f'seed_{seed}'/f'{c}-lora'


@app.function(image=image,gpu='L40S',cpu=4,memory=32768,timeout=24*60*60,max_containers=3,
              volumes={'/inputs':common.losses.base.inputs,'/results':common.losses.base.outputs},retries=0)
def train_head_backbone(run_id:str,arch:str,parallel_run:str,gcn_first_run:str,coefficient:float):
    import sys,json
    import pandas as pd
    import torch
    from types import SimpleNamespace
    from worker import prepare_runtime_source,write_json,write_csv
    if arch not in INPUTS:raise ValueError(arch)
    for value in [run_id,parallel_run,gcn_first_run]:
        if not value or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in value):raise ValueError('Invalid ID')
    if not torch.cuda.is_available() or 'L40S' not in torch.cuda.get_device_name(0):raise RuntimeError('Actual L40S required')
    repo=prepare_runtime_source('/repo','/tmp/afml-experiment2-source')
    sys.path[:0]=[str(repo/'src'),str(repo/'examples')]
    from train_adapter import code_provenance,file_hash
    from run_arxiv_experiments import Study
    from amortized_gnn_tta.training import software_versions
    from amortized_gnn_tta.adaptation.benchmark import execution_hardware
    expected=json.loads(Path('/experiment/original_CODE.json').read_text())['code']['source_hashes']
    if code_provenance()['source_hashes']!=expected:raise ValueError('Original source changed')
    bank=Path('/inputs/bank');index=json.loads((bank/'bank.json').read_text());manifest=json.loads((bank/'environment_manifest.json').read_text())
    if manifest.get('protocol')!='arxiv-transductive-v1' or manifest.get('dataset')!='ogbn-arxiv' or index['plan'].get('protocol')!='arxiv-formal-bank-v1':raise ValueError('Controlled-Arxiv bank required')
    bank_rows={r['id']:r for r in index['models']}
    refs={}
    for seed in SEEDS:
        folder=reference_folder('/results',parallel_run,gcn_first_run,arch,seed)
        r=json.loads((folder/'RESULT.json').read_text());recipe=json.loads((folder/'RECIPE.json').read_text())
        if r.get('loss_kind')!='kl' or r['conditioning']!=INPUTS[arch] or r['update']!='lora' or r['seed']!=seed or float(r['loss_coefficient'])!=coefficient:raise ValueError('Reference recipe mismatch')
        if recipe['max_epochs']!=100 or recipe['patience']!=10 or recipe['scheduler']!='CosineAnnealingLR' or recipe['lr_initial']!=.001 or recipe['lr_min']!=.00001:raise ValueError('Reference training setup differs')
        pins=json.loads((folder.parent/'study_context/SOURCE_PINS.json').read_text())
        if not pins or any(bank_rows.get(mid)!=row for mid,row in pins.items()):raise ValueError('Reference backbone bank differs')
        refs[seed]=dict(folder=str(folder),result_sha256=file_hash(folder/'RESULT.json'),
                       history_sha256=file_hash(folder/'history.csv'),result=r)
    root=Path('/results')/run_id/arch;root.mkdir(parents=True,exist_ok=False)
    write_json(root/'EXPERIMENT_PLAN.json',dict(experiment=2,architecture=arch,input=INPUTS[arch],seeds=SEEDS,
        update='head',references=refs,selection='validation only; epoch zero remains eligible',
        max_epochs=100,patience=10,scaling='same training-only coordinate RMS',KL_coefficient=coefficient,
        scheduler='cosine .001 toward .00001 over 100 epochs',hardware=execution_hardware('cuda'),
        software=software_versions(),code=code_provenance(),bank_sha256=file_hash(bank/'bank.json'),
        script_sha256=file_hash(Path(__file__)),reference_script_sha256=file_hash(Path(common.__file__)),
        final_test_scored=False))
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('highest')
    comparisons=[]
    for seed in SEEDS:
        seed_root=root/f'seed_{seed}';seed_root.mkdir()
        study=Study(SimpleNamespace(bank=bank,data_root=Path('/inputs/arxiv'),output_dir=seed_root/'study_context',stop_after=2))
        if study.plan.get('bank_kind')!='formal-real-v1' or study.plan.get('device')!='cuda':raise ValueError('Formal CUDA bank required')
        folder=seed_root/f'{INPUTS[arch]}-head'
        from contextlib import redirect_stdout
        with redirect_stdout(PrefixLog(sys.stdout,f'[E2 {arch} {INPUTS[arch]}-head seed={seed}] ')):
            result=fit_head(study,arch,INPUTS[arch],seed,folder,common.losses.base.outputs.commit,coefficient)
        # Same inputs require the same descriptor normalization, independently of update site.
        old_scale=torch.load(Path(refs[seed]['folder'])/'scaling.pt',map_location='cpu',weights_only=True)
        new_scale=torch.load(folder/'scaling.pt',map_location='cpu',weights_only=True)
        torch.testing.assert_close(old_scale['scale'],new_scale['scale'],rtol=1e-4,atol=1e-6)
        head_history=pd.read_csv(folder/'history.csv');lora_history=pd.read_csv(Path(refs[seed]['folder'])/'history.csv')
        lora=refs[seed]['result']
        comparisons.append(dict(architecture=arch,input=INPUTS[arch],seed=seed,
            head_selected_epoch=result['selected_epoch'],lora_selected_epoch=lora['selected_epoch'],
            head_selected_gain_pp=result['selected_gain_pp'],lora_selected_gain_pp=lora['selected_gain_pp'],
            paired_head_minus_lora_pp=result['selected_gain_pp']-lora['selected_gain_pp'],
            head_best_trained_gain_pp=float(head_history.val_gain.max()*100),
            lora_best_trained_gain_pp=float(lora_history.val_gain.max()*100),final_test_scored=False))
        write_csv(root/'head_vs_lora_by_seed.csv',comparisons)
        write_json(seed_root/'COMPLETE.json',dict(result=result,final_test_scored=False));common.losses.base.outputs.commit()
        del study;torch.cuda.empty_cache()
    frame=pd.DataFrame(comparisons)
    statistics={k:dict(mean=float(frame[k].mean()),sample_sd=float(frame[k].std(ddof=1))) for k in
        ['head_selected_gain_pp','lora_selected_gain_pp','paired_head_minus_lora_pp','head_best_trained_gain_pp','lora_best_trained_gain_pp']}
    write_json(root/'COMPARISON_SUMMARY.json',dict(architecture=arch,input=INPUTS[arch],seeds=SEEDS,statistics=statistics,final_test_scored=False))
    write_json(root/'COMPLETE.json',dict(results=comparisons,final_test_scored=False));common.losses.base.outputs.commit()
    return dict(architecture=arch,input=INPUTS[arch],statistics=statistics)


@app.function(image=image,cpu=1,memory=2048,timeout=24*60*60,retries=0)
def orchestrate(run_id:str,parallel_run:str,gcn_first_run:str,coefficient:float):
    # Remote parent keeps all workers alive if the local terminal disconnects.
    calls={a:train_head_backbone.spawn(run_id,a,parallel_run,gcn_first_run,coefficient) for a in INPUTS}
    results={};errors=[]
    for arch,call in calls.items():
        try:results[arch]=call.get();print(arch,results[arch],flush=True)
        except Exception as error:errors.append((arch,str(error)))
    if errors:raise RuntimeError(f'Partial results preserved: {errors}')
    return results


@app.local_entrypoint()
def main(run_id:str='',parallel_run:str='arxiv-kl-backbones-001',gcn_first_run:str='arxiv-new-losses-001',coefficient:float=1.):
    import math
    if not run_id:raise ValueError('Supply a unique --run-id')
    if not math.isfinite(coefficient) or coefficient<0:raise ValueError('Invalid coefficient')
    print(orchestrate.remote(run_id,parallel_run,gcn_first_run,coefficient))
