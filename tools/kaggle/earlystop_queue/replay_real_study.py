"""Independent reconstruction from saved source weights, factors and dense deltas.

No generator conditioning or optimization is repeated. Float tolerances and any
argmax differences are recorded, rather than assuming bitwise GPU determinism.
"""
import argparse,json
from pathlib import Path
import torch
from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.training import load_checkpoint
from amortized_gnn_tta.adaptation.checkpoints import load_generator_checkpoint,atomic_json
from amortized_gnn_tta.adaptation.generator import adapted_forward,extract_target_parameters
from amortized_gnn_tta.adaptation_artifacts import state_sha256
from amortized_gnn_tta.data.loaders import make_environment_loader


def run(a):
    torch.set_num_threads(1)
    bank=json.loads((a.bank/'bank.json').read_text());plan=bank['plan']
    manifest=json.loads((a.bank/'environment_manifest.json').read_text())
    loader=make_environment_loader(plan.get('dataset','ogbn-arxiv'),a.data_root,manifest)
    sources={};graphs={};generators={};rows=[]
    indexed={r['id']:r for r in bank['models']}
    tasks=[]
    for result in sorted(a.study.glob('*/RESULT.json')):
        fit=json.loads(result.read_text())
        for record in fit['records']:
            tasks.append((fit['name'],result.parent/record['artifact']))
    control=a.study/'CONDITIONING_CONTROL/COMPLETE.json'
    if control.exists():
        for fit in json.loads(control.read_text())['rows']:
            tasks.extend((fit['fit'],a.study/'CONDITIONING_CONTROL'/r['artifact']) for r in fit['records'])
    for fit,path in tasks:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        mid,eid=saved['source_id'],saved['environment_id']
        if mid not in sources:
            sources[mid]=load_checkpoint(a.bank/indexed[mid]['path'],build_model,device=a.device)[0]
        model=sources[mid]
        if state_sha256(model.state_dict())!=saved['source_state_sha256']:
            raise ValueError('Source identity differs from exported prediction')
        if eid not in graphs:
            graphs[eid]=loader(manifest,eid).unlabeled_input(eid).to(a.device)
        if fit not in generators:
            generators[fit]=load_generator_checkpoint(a.study/fit/'generator.pt',model=model,device=a.device)[0]
        generator=generators[fit]
        _,schema=extract_target_parameters(model,saved['scope'])
        if saved.get('factor_vector') is not None:
            product=generator.update_space.dense_delta(saved['factor_vector'].to(a.device))
            torch.testing.assert_close(product.cpu(),saved['delta'],rtol=0,atol=0)
        with torch.no_grad():
            replay=adapted_forward(model,graphs[eid],saved['delta'].to(a.device),schema)[saved['indices'].to(a.device)].cpu()
        reference=saved['prediction_logits']
        assert torch.equal(reference.argmax(-1),saved['predicted_classes'])
        torch.testing.assert_close(replay,reference,rtol=1e-4,atol=1e-5)
        rows.append(dict(artifact=str(path.relative_to(a.study)),max_abs_logit_difference=float((replay-reference).abs().max()),
                         class_differences=int((replay.argmax(-1)!=saved['predicted_classes']).sum()),
                         factors_verified=saved.get('factor_vector') is not None))
        if len(rows)%50==0: print('REPLAYED',len(rows),'of',len(tasks),flush=True)
    result=dict(status='complete',cases=len(rows),rows=rows,rtol=1e-4,atol=1e-5,
                prediction_class_differences=sum(r['class_differences'] for r in rows),
                final_test_scored=False,description='Independent dense-delta/prediction replay; actual saved factors only; no regenerated conditioning')
    atomic_json(a.study/'INDEPENDENT_REPLAY.json',result)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--device',default='cuda')
    print(json.dumps(run(p.parse_args())))
