"""KL-only controlled-Arxiv continuation: three backbones on separate L40S workers."""
import os
from pathlib import Path
import modal
import modal_loss_comparison_l40s as losses

HERE=Path(__file__).resolve().parent
app=modal.App('afml-kl-backbones-parallel')
image=losses.image.add_local_file(str(HERE/'modal_kl_backbones_parallel.py'),'/experiment/modal_kl_backbones_parallel.py')
SCHEDULE={'gcn':[202802,202803],'gat':[202801,202802,202803],'graphsage':[202801,202802,202803]}


class ArchitectureStudy:
    """Route the existing GCN-named fit wrapper to actual backbone checkpoints.

    Generator construction still uses the ACTUAL model, its configuration and
    parameter schema. This never converts GAT/SAGE weights into GCN weights.
    """
    def __init__(self,study,architecture):self.study=study;self.architecture=architecture
    def __getattr__(self,name):return getattr(self.study,name)
    def sources(self,architecture,role):return self.study.sources(self.architecture,role)
    def evaluate(self,architecture,*args,**kwargs):return self.study.evaluate(self.architecture,*args,**kwargs)


def run_fit(study,architecture,conditioning,seed,folder,commit,coefficient=1.,*,max_epochs=100,patience=10):
    import json
    import torch
    from unittest.mock import patch
    from worker import write_json
    from amortized_gnn_tta.adaptation import meta_trainer
    objective=losses.Objective('kl',coefficient)
    with patch.object(meta_trainer,'train_adaptation_generator',losses.adapted_trainer(meta_trainer,objective)):
        row=losses.base.fit_one(ArchitectureStudy(study,architecture),conditioning,seed,folder,commit,
                               max_epochs=max_epochs,patience=patience)
    folder=Path(folder)
    row.update(architecture=architecture,loss_kind='kl',loss_coefficient=coefficient)
    recipe=json.loads((folder/'RECIPE.json').read_text())
    recipe.update(architecture=architecture,loss_kind='kl',loss_coefficient=coefficient,
                  loss='CE + coefficient * KL(frozen || adapted) + delta_l2 * sum(dense_delta**2)',
                  KL_temperature=1.,KL_reduction='batchmean',KL_nodes='generator training nodes only')
    write_json(folder/'RECIPE.json',recipe);write_json(folder/'LOSS_RECIPE.json',recipe)
    payload=torch.load(folder/'best_generator.pt',map_location='cpu',weights_only=True)
    payload['provenance'].update(architecture=architecture,loss_kind='kl',loss_coefficient=coefficient,loss=recipe['loss'])
    torch.save(payload,folder/'best_generator.tmp');os.replace(folder/'best_generator.tmp',folder/'best_generator.pt')
    write_json(folder/'RESULT.json',row);write_json(folder/'STATUS.json',dict(state='complete',**row));commit()
    return row


@app.function(image=image,gpu='L40S',cpu=4,memory=32768,timeout=24*60*60,max_containers=3,
              volumes={'/inputs':losses.base.inputs,'/results':losses.base.outputs},retries=0)
def train_backbone(run_id:str,architecture:str,coefficient:float=1.):
    import json,sys
    import torch
    from types import SimpleNamespace
    from worker import prepare_runtime_source,write_json,write_csv
    if architecture not in SCHEDULE:raise ValueError(architecture)
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):raise ValueError('Invalid run-id')
    if not torch.cuda.is_available() or 'L40S' not in torch.cuda.get_device_name(0):raise RuntimeError('Actual L40S required')
    repo=prepare_runtime_source('/repo','/tmp/afml-kl-backbones-source')
    sys.path[:0]=[str(repo/'src'),str(repo/'examples')]
    from train_adapter import code_provenance,file_hash
    from run_arxiv_experiments import Study
    from amortized_gnn_tta.training import software_versions
    from amortized_gnn_tta.adaptation.benchmark import execution_hardware
    expected=json.loads(Path('/experiment/original_CODE.json').read_text())['code']['source_hashes']
    if code_provenance()['source_hashes']!=expected:raise ValueError('Original source hashes changed')
    bank=Path('/inputs/bank')
    manifest=json.loads((bank/'environment_manifest.json').read_text())
    indexed=json.loads((bank/'bank.json').read_text())
    if manifest.get('dataset')!='ogbn-arxiv' or manifest.get('protocol')!='arxiv-transductive-v1' or indexed['plan'].get('protocol')!='arxiv-formal-bank-v1':
        raise ValueError('Controlled-Arxiv bank required; temporal/country banks are not accepted')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('highest')
    root=Path('/results')/run_id/architecture;root.mkdir(parents=True,exist_ok=False)
    write_json(root/'CAMPAIGN.json',dict(architecture=architecture,seeds=SCHEDULE[architecture],inputs=list('ABCD'),
        loss='kl',coefficient=coefficient,max_epochs=100,patience=10,scaling='training-only coordinate RMS',
        lr_initial=.001,lr_min=.00001,schedule='cosine over 100 epochs',
        bank_sha256=file_hash(bank/'bank.json'),manifest_sha256=file_hash(bank/'environment_manifest.json'),
        code=code_provenance(),software=software_versions(),hardware=execution_hardware('cuda'),
        script_sha256=file_hash(Path(__file__)),loss_script_sha256=file_hash(Path(losses.__file__)),
        base_script_sha256=file_hash(Path(losses.base.__file__)),final_test_scored=False))
    all_rows=[]
    for seed in SCHEDULE[architecture]:
        seed_root=root/f'seed_{seed}';seed_root.mkdir()
        study=Study(SimpleNamespace(bank=bank,data_root=Path('/inputs/arxiv'),output_dir=seed_root/'study_context',stop_after=2))
        if study.plan.get('bank_kind')!='formal-real-v1' or study.plan.get('device')!='cuda':raise ValueError('Formal CUDA bank required')
        rows=[]
        for conditioning in 'ABCD':
            rows.append(run_fit(study,architecture,conditioning,seed,seed_root/f'{conditioning}-lora',losses.base.outputs.commit,coefficient))
            write_csv(seed_root/'summary.csv',rows);losses.base.outputs.commit()
        write_json(seed_root/'COMPLETE.json',dict(results=rows,final_test_scored=False))
        all_rows.extend(rows);write_csv(root/'summary.csv',all_rows);losses.base.outputs.commit()
        # Release cached source models before constructing the next seed's study.
        del study;torch.cuda.empty_cache()
    write_json(root/'COMPLETE.json',dict(results=all_rows,final_test_scored=False));losses.base.outputs.commit()
    return all_rows


@app.local_entrypoint()
def main(run_id:str='',coefficient:float=1.):
    import math
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):raise ValueError('Supply a unique valid --run-id')
    if not math.isfinite(coefficient) or coefficient<0:raise ValueError('Invalid KL coefficient')
    # Submit ALL three before waiting: three distinct workers/GPUs, not threads
    # mutating the trainer in one shared Python interpreter.
    calls={arch:train_backbone.spawn(run_id,arch,coefficient) for arch in SCHEDULE}
    errors=[]
    for arch,call in calls.items():
        print('Submitted',arch,'seeds',SCHEDULE[arch],'call',call.object_id,flush=True)
    for arch,call in calls.items():
        try:print(arch,call.get())
        except Exception as error:errors.append((arch,str(error)));print('FAILED',arch,str(error),flush=True)
    if errors:raise RuntimeError(f'Backbone workers failed; outputs preserved: {errors}')
