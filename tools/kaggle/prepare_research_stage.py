"""Prepare a private next-stage job, gated on verified upstream evidence."""
import argparse,ast,base64,hashlib,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=['extend','final','natural','diagnostic'],required=True)
p.add_argument('--dataset',choices=['ogbn-arxiv','ogbn-arxiv-temporal','pair-align-mag-country'],required=True)
p.add_argument('--bank-input',required=True);p.add_argument('--study-input',required=True)
p.add_argument('--bank-proof',type=Path,required=True);p.add_argument('--study-proof',type=Path,required=True)
p.add_argument('--architecture',choices=['gcn','gat','graphsage'],default='gcn')
p.add_argument('--slug',required=True);p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--phase-gate',type=Path)
p.add_argument('--runtime-seconds',type=int,help='Explicit remote job limit; saved in the manifest and passed to the worker')
a=p.parse_args()
runtime=a.runtime_seconds or (32400 if a.mode in {'extend','natural'} else 14400 if a.mode=='final' and a.dataset=='ogbn-arxiv' else 7200 if a.mode=='final' else 3600)
if not 2400<=runtime<=43200:raise ValueError('Runtime must leave room for evidence packaging within the documented session limit')
bank=json.loads(a.bank_proof.read_text());study=json.loads(a.study_proof.read_text())
if bank.get('status')!='complete' or len(bank.get('source_fits',[]))!=90:raise ValueError('Complete independently verified bank required')
if study.get('status')!='complete' or study.get('study_source_sha256')!='26a40737a7c86e6a0d6d117e4ac5b2ee5c8fc6b21b96c81a3ac0c14dd3fbea56':raise ValueError('Verified matching immutable study required')
if a.mode=='natural':
    if not a.phase_gate or not json.loads(a.phase_gate.read_text()).get('experiment_4_complete'):raise ValueError('Complete controlled final audit must precede natural-shift studies')
if a.mode=='final' and not study.get('frozen_design'):raise ValueError('Frozen design required before final labels')
if a.mode=='diagnostic' and (not a.phase_gate or not json.loads(a.phase_gate.read_text()).get('experiment_5_complete')):raise ValueError('Natural studies precede the final diagnostic milestone')
raw=(ROOT.parent/'amortized-gnn-tta-generator-integration/runs/formal-study-kaggle-v1/source.zip').read_bytes()
sha=hashlib.sha256(raw).hexdigest()
if sha!=study['study_source_sha256']:raise ValueError('Source ZIP changed')
headers=dict(STUDY_ARCHIVE=base64.b85encode(raw).decode(),STUDY_SHA256=sha,
             REPLAY_CODE=(ROOT/'replay_real_study.py').read_text(),FINAL_RUNNER=(ROOT/'evaluate_final_shard.py').read_text(),DIAGNOSTIC_RUNNER=(ROOT/'diagnose_real_audit.py').read_text(),
             MODE=a.mode,DATASET=a.dataset,ARCHITECTURE=a.architecture,STUDY_INPUT=a.study_input,STAGE_TIMEOUT=runtime-1800)
common=(ROOT/'gcn_study_driver.py').read_text().split('\ntry:\n',1)[0]
script=''.join(k+' = '+repr(v)+'\n' for k,v in headers.items())+common+'\n'+(ROOT/'research_stage_driver.py').read_text();ast.parse(script)
a.output_dir.mkdir(parents=True,exist_ok=False);(a.output_dir/'stage.py').write_text(script)
m=json.loads((ROOT/'kaggle-gcn-study-20261108/server-version-1/kernel-metadata.json').read_text())
m.update(id='deeptanshukumar/'+a.slug,title=a.slug.replace('-',' '),code_file='stage.py',is_private=True,kernel_sources=list(dict.fromkeys([a.bank_input,a.study_input])),docker_image_pinning_type='original');m.pop('id_no',None)
(a.output_dir/'kernel-metadata.json').write_text(json.dumps(m,indent=2)+'\n')
(a.output_dir/'LOCAL_MANIFEST.json').write_text(json.dumps(dict(mode=a.mode,dataset=a.dataset,architecture=a.architecture if a.mode=='final' else None,
    study_source_sha256=sha,script_sha256=hashlib.sha256(script.encode()).hexdigest(),bank_input=a.bank_input,study_input=a.study_input,is_private=True,timeout_seconds=runtime,stage_timeout_seconds=runtime-1800),indent=2)+'\n')
print(json.dumps(dict(prepared=m['id'],mode=a.mode,is_private=True)))
