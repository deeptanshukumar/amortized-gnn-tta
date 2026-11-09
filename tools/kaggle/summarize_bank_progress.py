"""Consolidate verified formal-bank checkpoints without counting resumed fits twice."""
from pathlib import Path
import csv,json,statistics
root=Path(__file__).resolve().parent
chosen={}
for proof_path in root.glob('kaggle-*/download-version-*/LOCAL_REPLAY_VERIFICATION.json'):
    proof=json.loads(proof_path.read_text())
    if proof.get('status')!='complete':continue
    evidence=proof_path.parent/'evidence'
    identity=json.loads((evidence/'formal-bank/IDENTITY.json').read_text())
    setup=identity['plan'].get('dataset','ogbn-arxiv')
    if setup in chosen:
        previous=chosen[setup][1];old={r['id']:r['checkpoint_sha256'] for r in previous['source_fits']}
        for row in proof['source_fits']:
            if row['id'] in old and old[row['id']]!=row['checkpoint_sha256']:raise ValueError('Checkpoint replacement across runs')
    if setup not in chosen or len(proof['source_fits'])>len(chosen[setup][1]['source_fits']):chosen[setup]=(proof_path,proof,identity)
rows=[];summary=[]
for setup,(path,proof,identity) in sorted(chosen.items()):
    for row in proof['source_fits']:
        rows.append(dict(dataset=setup,architecture=row['architecture'],group=row['group'],role=row['role'],seed=row['seed'],
                         selected_epoch=row['selected_epoch'],fit_seconds=row['fit_seconds'],cuda_peak_allocated_bytes=row['cuda_peak_allocated_bytes'],
                         cuda_peak_reserved_bytes=row['cuda_peak_reserved_bytes'],checkpoint_sha256=row['checkpoint_sha256'],state_sha256=row['state_sha256'],
                         evidence=str(path.relative_to(root))))
    for arch in ('gcn','gat','graphsage'):
        fits=[r for r in proof['source_fits'] if r['architecture']==arch]
        summary.append(dict(dataset=setup,architecture=arch,verified_models=len(fits),expected_models=30,
                            fit_seconds=sum(r['fit_seconds'] for r in fits),mean_fit_seconds=statistics.mean(r['fit_seconds'] for r in fits) if fits else None,
                            max_cuda_allocated_bytes=max((r['cuda_peak_allocated_bytes'] for r in fits),default=None)))
with (root/'FORMAL_SOURCE_FITS.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
(root/'FORMAL_BANK_SUMMARY.json').write_text(json.dumps(dict(verified_models=len(rows),expected_models=270,summary=summary,final_target_labels_scored=False),indent=2)+'\n')
lines=['# Verified formal source banks','',f'Verified unique checkpoints: **{len(rows)}/270**. Final target labels scored: **No**.','',
       '| Source protocol | Architecture | Verified / expected | Summed fit seconds | Mean fit seconds |','|---|---|---:|---:|---:|']
for s in summary:lines.append(f"| {s['dataset']} | {s['architecture']} | {s['verified_models']}/30 | {s['fit_seconds']:.2f} | {s['mean_fit_seconds']:.2f} |" if s['mean_fit_seconds'] is not None else f"| {s['dataset']} | {s['architecture']} | 0/30 | 0 | pending |")
lines+=['','Times are recorded source-fit/checkpoint-validation wall times on the actual GPU backend, excluding dataset download, runtime startup and broader research search. Checkpoints reused by later jobs are counted once. Fixed budgets and late selected epochs do not establish convergence. Source-only diversity audits are separate evidence, not a universal sufficiency guarantee.']
(root/'FORMAL_BANK_REPORT.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(dict(verified_unique_models=len(rows),expected=270)))
