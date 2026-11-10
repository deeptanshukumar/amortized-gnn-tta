"""Matched loss ablation; imports existing L40S runner without editing it."""
import os
from pathlib import Path
import modal
HERE=Path(__file__).resolve().parent
REPO=Path(os.environ.get('AFML_REPO',str(HERE.parent)))
# Reuse the complete image, artifacts, optimizer schedule and fit/export machinery.
import modal_input_comparison_l40s as base
app=modal.App('afml-loss-comparison-l40s')
image=(base.image
       .add_local_file(str(HERE/'modal_input_comparison_l40s.py'),'/experiment/modal_input_comparison_l40s.py')
       .add_local_file(str(HERE/'modal_loss_comparison_l40s.py'),'/experiment/modal_loss_comparison_l40s.py'))

class Objective:
    def __init__(self,kind,coefficient=1.):
        self.kind=kind;self.coefficient=coefficient;self.rows=[];self.last={}
    def __call__(self,logits,labels,model,graph,indices):
        import torch
        import torch.nn.functional as F
        ce=F.cross_entropy(logits,labels)
        with torch.no_grad():
            frozen=model(graph.x,graph.edge_index)[indices].detach()
            frozen_ce=F.cross_entropy(frozen,labels)
        excess=torch.relu(ce-frozen_ce)
        # KL(frozen || adapted), averaged per node, summed over the 40 classes.
        kl=F.kl_div(F.log_softmax(logits,dim=-1),F.softmax(frozen,dim=-1),reduction='batchmean')
        auxiliary=excess if self.kind=='excess_ce' else kl if self.kind=='kl' else ce.new_zeros(())
        self.last=dict(adapted_ce=float(ce.detach()),frozen_ce=float(frozen_ce),
                       auxiliary_raw=float(auxiliary.detach()),
                       auxiliary_weighted=float((self.coefficient*auxiliary).detach()),
                       excess_active=float(ce.detach()>frozen_ce))
        self.rows.append(self.last)
        return ce+self.coefficient*auxiliary


def adapted_trainer(module,objective):
    """Clone the original function in memory; replace only CE construction/logging.

    No on-disk source changes, global functional monkeypatches or replacement loop.
    """
    import ast,inspect,textwrap
    tree=ast.parse(textwrap.dedent(inspect.getsource(inspect.unwrap(module.train_adaptation_generator))))
    replacements=0;logging=0
    class Modify(ast.NodeTransformer):
        def visit_Assign(self,node):
            nonlocal replacements
            if any(isinstance(t,ast.Name) and t.id=='ce_loss' for t in node.targets):
                assert ast.unparse(node.value)=='F.cross_entropy(logits[train_idx], train_y)'
                node.value=ast.parse('_loss_objective(logits[train_idx], train_y, model, unlabeled, train_idx)',mode='eval').body
                replacements+=1
            return self.generic_visit(node)
        def visit_Call(self,node):
            nonlocal logging
            if ast.unparse(node)=='ce_losses.append(ce_loss.item())':
                node.args=[ast.parse("_loss_objective.last['adapted_ce']",mode='eval').body];logging+=1
            return self.generic_visit(node)
    tree=Modify().visit(tree);ast.fix_missing_locations(tree)
    if replacements!=1 or logging!=1:raise RuntimeError('Unexpected original trainer structure')
    namespace=dict(module.__dict__);namespace['_loss_objective']=objective
    # Existing fit_one installs its per-case recorder dynamically.
    namespace['evaluate_adaptation_pair']=lambda *a,**kw:module.evaluate_adaptation_pair(*a,**kw)
    exec(compile(tree,'<AFML loss-only runtime adapter>','exec'),namespace)
    cloned=namespace['train_adaptation_generator']
    def run(*args,**kwargs):
        callback=kwargs.get('checkpoint_callback')
        def enriched(state,history):
            if history:
                import numpy as np
                if len(objective.rows)!=126:raise RuntimeError('Expected 126 training pair losses per epoch')
                row=history[-1]
                for key in objective.last:row[key]=float(np.mean([r[key] for r in objective.rows]))
                row['loss_kind']=objective.kind;row['loss_coefficient']=objective.coefficient
                objective.rows.clear()
            if callback:callback(state,history)
        kwargs['checkpoint_callback']=enriched
        return cloned(*args,**kwargs)
    return run


@app.function(image=image,gpu='L40S',cpu=4,memory=32768,timeout=24*60*60,
              volumes={'/inputs':base.inputs,'/results':base.outputs},retries=0)
def train_loss_seed(run_id:str,kind:str,seed:int,coefficient:float):
    import sys,json
    import torch
    from types import SimpleNamespace
    from unittest.mock import patch
    from worker import prepare_runtime_source,write_json,write_csv
    if kind not in {'original','excess_ce','kl'}:raise ValueError(kind)
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):raise ValueError('Invalid run-id')
    if not torch.cuda.is_available() or 'L40S' not in torch.cuda.get_device_name(0):raise RuntimeError('L40S required')
    repo=prepare_runtime_source('/repo','/tmp/afml-loss-comparison-source')
    sys.path[:0]=[str(repo/'src'),str(repo/'examples')]
    from run_arxiv_experiments import Study
    from train_adapter import code_provenance,file_hash
    from amortized_gnn_tta.adaptation import meta_trainer
    from amortized_gnn_tta.training import software_versions
    from amortized_gnn_tta.adaptation.benchmark import execution_hardware
    expected=json.loads(Path('/experiment/original_CODE.json').read_text())['code']['source_hashes']
    if code_provenance()['source_hashes']!=expected:raise ValueError('Original source changed')
    root=Path('/results')/run_id/kind/f'seed_{seed}';root.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('highest')
    study=Study(SimpleNamespace(bank=Path('/inputs/bank'),data_root=Path('/inputs/arxiv'),output_dir=root/'study_context',stop_after=2))
    if study.plan.get('bank_kind')!='formal-real-v1' or study.plan.get('device')!='cuda':raise ValueError('Formal CUDA bank required')
    write_json(root/'LOSS_RECIPE.json',dict(kind=kind,coefficient=coefficient,
        objective='CE + coefficient*relu(CE-frozen_CE)' if kind=='excess_ce' else
                  'CE + coefficient*KL(frozen||adapted)' if kind=='kl' else 'CE',
        update_penalty=study.plan['delta_l2'],node_pool='generator-training nodes only',temperature=1.,
        frozen_reference='same pretrained GNN, same corrupted graph, no dropout, detached',
        code=code_provenance(),hardware=execution_hardware('cuda'),software=software_versions(),
        script_sha256=file_hash(Path(__file__)),base_script_sha256=file_hash(Path(base.__file__)),
        final_test_scored=False))
    rows=[]
    for conditioning in 'ABCD':
        objective=Objective(kind,coefficient)
        with patch.object(meta_trainer,'train_adaptation_generator',adapted_trainer(meta_trainer,objective)):
            row=base.fit_one(study,conditioning,seed,root/f'{conditioning}-lora',base.outputs.commit)
        row.update(loss_kind=kind,loss_coefficient=coefficient)
        folder=root/f'{conditioning}-lora'
        loss_recipe=json.loads((root/'LOSS_RECIPE.json').read_text())
        write_json(folder/'LOSS_RECIPE.json',loss_recipe)
        # Correct the reused runner's default objective annotation on NEW artifacts.
        fit_recipe=json.loads((folder/'RECIPE.json').read_text())
        fit_recipe['loss']=loss_recipe['objective']+' + delta_l2 * sum(dense_delta**2)'
        fit_recipe['loss_coefficient']=coefficient
        write_json(folder/'RECIPE.json',fit_recipe)
        checkpoint=torch.load(folder/'best_generator.pt',map_location='cpu',weights_only=True)
        checkpoint['provenance']['loss']=fit_recipe['loss']
        checkpoint['provenance']['loss_kind']=kind
        checkpoint['provenance']['loss_coefficient']=coefficient
        torch.save(checkpoint,folder/'best_generator.tmp')
        os.replace(folder/'best_generator.tmp',folder/'best_generator.pt')
        write_json(root/f'{conditioning}-lora'/'RESULT.json',row)
        rows.append(row);write_csv(root/'summary.csv',rows);base.outputs.commit()
    write_json(root/'COMPLETE.json',dict(results=rows,final_test_scored=False));base.outputs.commit()
    return rows


@app.local_entrypoint()
def main(run_id:str='',seeds:str='202801,202802,202803',losses:str='original,excess_ce,kl',coefficient:float=1.):
    import math
    if not run_id:raise ValueError('Supply a unique run-id')
    seed_list=[int(s.strip()) for s in seeds.split(',')];kinds=[s.strip() for s in losses.split(',')]
    if len(set(seed_list))!=len(seed_list) or any(not 0<=s<2**32 for s in seed_list):raise ValueError('Invalid seeds')
    if len(set(kinds))!=len(kinds) or not set(kinds)<={'original','excess_ce','kl'}:raise ValueError('Invalid losses')
    if not math.isfinite(coefficient) or coefficient<0:raise ValueError('Invalid coefficient')
    for seed in seed_list:
        for kind in kinds:print(train_loss_seed.remote(run_id,kind,seed,coefficient))
