"""Verify downloaded private preflight outputs and replay captured CUDA artifacts.

Run with the project's existing research Python; no remote calls or training.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--download',type=Path,required=True)
    p.add_argument('--local-manifest',type=Path,required=True)
    p.add_argument('--expected-models',type=int,default=1)
    p.add_argument('--bank-only',action='store_true')
    a=p.parse_args()
    statuses=list(a.download.rglob('STATUS.json'))
    if len(statuses)!=1:
        raise ValueError('Expected exactly one execution status')
    evidence=statuses[0].parent
    status=json.loads(statuses[0].read_text())
    if status['state']!='complete':
        raise ValueError('Remote job did not complete: '+json.dumps(status))
    bundle=evidence/'BUNDLE.zip'
    if bundle.exists():
        with zipfile.ZipFile(bundle) as z:
            for info in z.infolist():
                rel=Path(info.filename)
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('Unsafe evidence bundle member')
                target=evidence/rel
                raw=z.read(info.filename)
                if target.exists() and target.read_bytes()!=raw:
                    raise ValueError('Downloaded and bundled output differ')
                target.parent.mkdir(parents=True,exist_ok=True)
                if not target.exists(): target.write_bytes(raw)
    expected=json.loads(a.local_manifest.read_text())['source_zip_sha256']
    if digest(evidence/'SOURCE.zip')!=expected:
        raise ValueError('Downloaded source archive mismatch')
    hashes=json.loads((evidence/'OUTPUT_HASHES.json').read_text())
    for name, value in hashes.items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe output hash member')
        if digest(evidence/path)!=value:
            raise ValueError('Downloaded output hash mismatch: '+name)
    source=a.download/'replay-source'
    source.mkdir(exist_ok=True)
    with zipfile.ZipFile(evidence/'SOURCE.zip') as z:
        manifest=json.loads(z.read('amortized-gnn-tta/colab_bundle.json'))
        for name,value in manifest['files_sha256'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe source member')
            raw=z.read('amortized-gnn-tta/'+name)
            if hashlib.sha256(raw).hexdigest()!=value:
                raise ValueError('Captured source hash mismatch')
            target=source/name; target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists() and target.read_bytes()!=raw: raise ValueError('Replay source changed')
            if not target.exists(): target.write_bytes(raw)
    sys.path.insert(0,str(source/'src'))
    import torch
    from torch_geometric.data import Data
    from amortized_gnn_tta.models import build_model
    from amortized_gnn_tta.adaptation.lora_generator import LoRAUpdateSpace
    from amortized_gnn_tta.adaptation.generator import adapted_forward
    from amortized_gnn_tta.adaptation_artifacts import state_sha256
    torch.set_num_threads(1)
    replays=[]
    files=sorted((evidence/'factor-replays').glob('*.pt'))
    if not a.bank_only and len(files)!=6:
        raise ValueError('Expected all three architectures and both generator families')
    for path in files:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=build_model(saved['model_spec']); model.load_state_dict(saved['source_state']); model.eval()
        space=LoRAUpdateSpace(**saved['lora_space'])
        dense=space.dense_delta(saved['factors'])
        torch.testing.assert_close(dense,saved['delta'],rtol=1e-4,atol=1e-5)
        graph=Data(x=saved['x'],edge_index=saved['edge_index'],num_nodes=len(saved['x']))
        with torch.no_grad(): logits=adapted_forward(model,graph,saved['delta'],saved['schema'])
        torch.testing.assert_close(logits,saved['logits'],rtol=1e-4,atol=1e-5)
        difference=int((logits.argmax(-1)!=saved['logits'].argmax(-1)).sum())
        if difference:
            raise ValueError('Saved prediction-class replay mismatch')
        replays.append(dict(file=path.name,factor_delta_max_abs_difference=float((dense-saved['delta']).abs().max()),
                            logit_max_abs_difference=float((logits-saved['logits']).abs().max()),
                            prediction_class_differences=difference,selection=saved['selection']))
    bank=evidence/'formal-bank'; manifest=json.loads((bank/'bank.json').read_text())
    if len(manifest['models'])!=a.expected_models or manifest['status']!=('complete' if a.expected_models==manifest['expected_models'] else 'partial'):
        raise ValueError('Unexpected source-bank completion claim')
    identity=json.loads((bank/'IDENTITY.json').read_text())
    from amortized_gnn_tta.data.source_bank import json_sha256
    identity_hash=json_sha256(identity)
    jobs={j['id']:j for j in identity['jobs']}
    for row in manifest['models']:
        if row['job']!=jobs[row['id']] or row['identity_sha256']!=identity_hash:
            raise ValueError('Source model recipe identity mismatch')
        checkpoint=bank/row['path']
        payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if (digest(checkpoint)!=row['checkpoint_sha256'] or state_sha256(payload['state_dict'])!=row['state_sha256']
                or payload['model_spec']!=row['job']['model_spec']
                or payload['training_config']!=row['job']['training_config']
                or payload['source']['formal_group']!=identity['groups'][row['group']]):
            raise ValueError('Downloaded source checkpoint mismatch')
    report=dict(status='complete',source_zip_sha256=expected,verified_output_files=len(hashes),
                replays=replays,source_fit=row,source_fits=manifest['models'],hardware=json.loads((evidence/'HARDWARE.json').read_text()),
                final_test_scored=False,remote_versioned_resume_verified=bool(status.get('remote_versioned_resume_verified')))
    (a.download/'LOCAL_REPLAY_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='complete',factor_prediction_replays=len(replays),source_fit_seconds=row['fit_seconds'],
                         cuda_peak_allocated_bytes=row['cuda_peak_allocated_bytes'],final_test_scored=False)))

if __name__=='__main__': main()
