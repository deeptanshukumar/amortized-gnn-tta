"""Consolidate three independently verified final shards; never select methods."""
import argparse,csv,hashlib,itertools,json,statistics
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--downloads',nargs=3,type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--dataset',choices=['ogbn-arxiv','ogbn-arxiv-temporal','pair-align-mag-country'],required=True)
a=p.parse_args();all_rows=[];plans=[];provenance=[];architectures=[];environment_hashes=[]
for download in a.downloads:
    proof=json.loads((download/'LOCAL_STUDY_VERIFICATION.json').read_text())
    if proof.get('status')!='complete' or proof.get('mode')!='final':raise ValueError('Independently verified final shard required')
    study=download/proof['study_relative_path'];final=study/'FINAL'
    inputs=json.loads((study/'INPUTS.json').read_text())
    if inputs['plan'].get('dataset','ogbn-arxiv')!=a.dataset:raise ValueError('Dataset mismatch')
    plan=json.loads((final/'EVALUATION_PLAN.json').read_text());plans.append(plan)
    environment_hashes.append(hashlib.sha256((final/'ENVIRONMENTS.json').read_bytes()).hexdigest())
    architectures.append(plan['architecture_shard'])
    rows=json.loads((final/'SCORES.json').read_text())['rows']
    if any(r['architecture']!=plan['architecture_shard'] for r in rows):raise ValueError('Architecture contamination')
    all_rows.extend(rows)
    provenance.append(dict(download=str(download.resolve()),proof_sha256=hashlib.sha256((download/'LOCAL_STUDY_VERIFICATION.json').read_bytes()).hexdigest(),cases=len(rows)))
if set(architectures)!={'gcn','gat','graphsage'} or len(set(architectures))!=3:raise ValueError('All three architecture shards required')
if len(set(environment_hashes))!=1:raise ValueError('Final environments differ across architecture shards')
for name in ('design_sha256','code','software','hardware','audit_export_runner_sha256','replay_tolerances','timing_warmups','timing_repeats'):
    if any(plan[name]!=plans[0][name] for plan in plans):raise ValueError('Mixed final protocol: '+name)
keys={(r['architecture'],r['model_id'],r['environment_id'],r['method']) for r in all_rows}
if len(keys)!=len(all_rows):raise ValueError('Duplicate final cases')
summary=[]
for arch,method in sorted({(r['architecture'],r['method']) for r in all_rows}):
    rows=[r for r in all_rows if r['architecture']==arch and r['method']==method]
    summary.append(dict(architecture=arch,method=method,cases=len(rows),mean_accuracy=statistics.mean(r['accuracy'] for r in rows),mean_gain=statistics.mean(r['gain'] for r in rows),harm_fraction=statistics.mean(r['gain']<0 for r in rows),effective_identity_cases=sum(r.get('effective_identity_update') is True for r in rows)))
a.output_dir.mkdir(parents=True,exist_ok=False)
with (a.output_dir/'ALL_FINAL_SCORES.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(all_rows[0]));writer.writeheader();writer.writerows(all_rows)
lines=['# Verified final evaluation','',f'Dataset: `{a.dataset}`. All three architecture shards independently verified.','',f'Total method/checkpoint/environment observations: {len(all_rows)}.','', '| Architecture | Method | Cases | Mean accuracy | Mean gain (pp) | Harmful cases | Exported zero updates |','|---|---|---:|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['architecture']} | {r['method']} | {r['cases']} | {100*r['mean_accuracy']:.3f}% | {100*r['mean_gain']:+.3f} | {100*r['harm_fraction']:.1f}% | {r['effective_identity_cases']} |")
lines+=['','Means describe the predeclared cases; shared underlying graphs are not independent datasets. Generator seeds stay separately labeled. Zero-update counts apply to exported weight-update artifacts; frozen and feature-only methods have no such update artifact. Exact identity updates cannot establish adaptation gains; small prediction differences can arise from floating arithmetic. Both constant-teacher construction and practice-fixed fitting use permitted offline practice labels, with no final labels in fitting. Timing and memory records remain in each shard. GTrans is the declared feature-only variant, with no saved feature-update replay. No settings are selected from these final scores.','']
(a.output_dir/'FINAL_REPORT.md').write_text('\n'.join(lines))
gate=dict(status='complete',dataset=a.dataset,architectures=sorted(architectures),cases=len(all_rows),design_sha256=plans[0]['design_sha256'],audit_export_runner_sha256=plans[0]['audit_export_runner_sha256'],summary=summary,provenance=provenance)
if a.dataset=='ogbn-arxiv':gate['experiment_4_complete']=True
else:gate['natural_final_complete']=True
(a.output_dir/'PHASE_GATE.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps({k:v for k,v in gate.items() if k not in {'summary','provenance'}}))
