"""Read-only diagnostic of saved practice teachers and initial generator states.
No graph training, no final labels, and no changes to the running study.
"""
from pathlib import Path
import collections,csv,json,math
import torch

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
study=ROOT/'kaggle-gcn-study-20261108/download-version-1/evidence/controlled-study'
t=torch.load(study/'teachers-gcn-A-head.pt',map_location='cpu',weights_only=True)
r=t['records'];groups=collections.defaultdict(list)
for row in r:groups[row['environment_id']].append(row)
fields=['global_mean','global_std','node_norm_mean','node_norm_std','zero_fraction','feature_max','feature_min','degree_mean','degree_std','degree_max_over_mean','isolated_fraction','edges_per_node','mean_edge_energy','edge_distance_mean','edge_distance_std','distance_to_center_mean','distance_to_center_std']
name=lambda i:('feature_mean_'+str(i)) if i<128 else ('feature_std_'+str(i-128)) if i<256 else fields[i-256]
c=torch.load(study/'gcn-A-head-hypernetwork-202801/generator.pt',map_location='cpu',weights_only=True)
s=c['state_dict'];rows=[]
for eid,rs in groups.items():
    descriptors=torch.stack([x['descriptor'] for x in rs]);updates=torch.stack([x['target_delta'] for x in rs]);weights=torch.stack([x['head_weights'] for x in rs]);norms=updates.norm(dim=1)
    unit=updates/norms.clamp_min(1e-30)[:,None];cos=unit@unit.T;off=~torch.eye(len(rs),dtype=torch.bool)
    d=descriptors[0];hidden=torch.relu(torch.nn.functional.linear(d,s['net.0.weight'],s['net.0.bias']))
    idx=int(d.abs().argmax());energy_fraction=float(d[idx].square()/d.square().sum())
    rows.append(dict(environment=eid,models=len(rs),max_model_descriptor_difference=float((descriptors-descriptors[0]).abs().max()),largest_descriptor_field=name(idx),largest_descriptor_abs=float(d[idx].abs()),largest_field_fraction_input_squared_norm=energy_fraction,
        initial_hidden_l1=float(hidden.abs().sum()),initial_hidden_l2=float(hidden.norm()),first_adam_step_output_coordinate_bound_ignoring_epsilon=0.001*(float(hidden.abs().sum())+1),
        teacher_delta_mean_norm=float(norms.mean()),source_head_mean_norm=float(weights.norm(dim=1).mean()),teacher_relative_mean_norm=float((norms/weights.norm(dim=1)).mean()),
        teacher_mean_pairwise_cosine=float(cos[off].mean()),norm_mean_update_over_mean_norm=float(updates.mean(dim=0).norm()/norms.mean()),
        teacher_mean_train_gain_pp=100*sum(x['oracle_acc']-x['baseline_acc'] for x in rs)/len(rs)))
history=list(csv.DictReader((ROOT/'verified-design-20261108/TRAINING_HISTORY.csv').open()))
result=dict(status='complete',labels_access='Only previously saved practice training scores; no final labels read',conditioning_A_is_model_independent_by_construction=True,generator_contains_descriptor_standardization=False,
    environment_diagnostics=rows,first_epoch_history=[{k:v for k,v in x.items() if k in {'fit','delta_norm','ce_loss','val_gain'}} for x in history if x['epoch']=='1'],
    first_step_bound_interpretation='Analytic magnitude bound for a zero-initialized output layer after one Adam step at lr.001, holding the initial hidden activation fixed and ignoring Adam epsilon/weight decay. It is not a measured first update or a proven causal effect.')
(Path(__file__).parent/'SAVED_SIGNAL_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
