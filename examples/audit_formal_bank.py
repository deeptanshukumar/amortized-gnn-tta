"""Label-free formal-bank diversity on a saved source-only probe.

Probe size/seed are explicit arguments; no diversity threshold is imposed.
"""
import argparse
from itertools import combinations
import json
from pathlib import Path

import torch
from amortized_gnn_tta.adaptation.checkpoints import atomic_json
from amortized_gnn_tta.bank_diagnostics import linear_cka, describe
from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.training import load_checkpoint
from amortized_gnn_tta.data.source_bank import tensor_sha256
from prepare_formal_bank import execute
from prepare_arxiv_bank import load_bank_inputs
from train_adapter import file_hash


def audit(a):
    # This checks every source identity/checkpoint even when the index was rebuilt.
    a.output_dir = a.bank
    a.fit = False
    execute(a)
    bank = json.loads((a.bank/'bank.json').read_text())
    bank_sha256 = file_hash(a.bank/'bank.json')
    plan = bank['plan']
    if plan['device'] != 'cuda' and not getattr(a, 'fixture', False):
        raise ValueError('Real full-graph diversity inference must run on GPU')
    _, view, native = load_bank_inputs(a, plan)
    pool = view.train_indices
    rng = torch.Generator().manual_seed(a.probe_seed)
    probe = pool[torch.randperm(len(pool), generator=rng)[:a.probe_nodes]].sort().values
    if len(probe) < 2:
        raise ValueError('Diversity needs at least two probe nodes')
    graph = view.graph.clone().to(plan['device'])
    by_architecture = {}
    for architecture in plan['architectures']:
        rows = [r for r in bank['models'] if r['architecture'] == architecture]
        weights, predictions, embeddings = [], [], []
        for row in rows:
            if file_hash(a.bank/row['path']) != row['checkpoint_sha256']:
                raise ValueError('Checkpoint changed during diversity audit')
            model, _ = load_checkpoint(a.bank/row['path'], build_model, device=plan['device'])
            weights.append(torch.cat([p.detach().cpu().double().flatten() for _, p in sorted(model.named_parameters())]))
            with torch.inference_mode():
                predictions.append(model(graph.x, graph.edge_index)[probe].softmax(-1).cpu())
                embeddings.append(model.encode(graph.x, graph.edge_index)[probe].cpu())
            del model
        pairs = []
        for i, j in combinations(range(len(rows)), 2):
            left, right = weights[i], weights[j]
            scale = float((left.norm()+right.norm())/2)
            pairs.append(dict(first=rows[i]['id'], second=rows[j]['id'],
                              same_supervision_group=rows[i]['group']==rows[j]['group'],
                              relative_weight_l2=float((left-right).norm())/scale if scale else None,
                              exact_parameter_duplicate=torch.equal(left, right),
                              prediction_disagreement=float((predictions[i].argmax(-1)!=predictions[j].argmax(-1)).float().mean()),
                              probability_rmse=float((predictions[i]-predictions[j]).square().mean().sqrt()),
                              hidden_linear_cka=linear_cka(embeddings[i], embeddings[j])))
        metrics=('relative_weight_l2','prediction_disagreement','probability_rmse','hidden_linear_cka')
        by_architecture[architecture]=dict(models=len(rows), pairs=pairs,
                    summary={m:describe([p[m] for p in pairs if p[m] is not None]) for m in metrics})
    if file_hash(a.bank/'bank.json') != bank_sha256:
        raise ValueError('Bank changed during diversity audit; preserve it and rerun')
    result=dict(bank_sha256=bank_sha256, probe_indices=probe.tolist(),
                probe_native_ids=native[probe].tolist(), probe_sha256=tensor_sha256(probe),
                probe_seed=a.probe_seed, requested_probe_nodes=a.probe_nodes,
                final_test_access=False, architectures=by_architecture,
                interpretation='Source-only label-free probe; descriptive variation, no sufficient-diversity or convergence threshold. Raw weight distance has neuron-permutation ambiguity.')
    atomic_json(a.bank/'DIVERSITY.json', result)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--probe-nodes',type=int,required=True)
    p.add_argument('--probe-seed',type=int,required=True)
    a=p.parse_args()
    if a.probe_nodes < 2 or not 0 <= a.probe_seed < 2**32:
        p.error('Need >=2 probe nodes and a seed in [0,2**32)')
    audit(a)
