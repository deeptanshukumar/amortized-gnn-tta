"""Resumable formal real-data banks. Default prepares only; fitting requires CUDA.

Completed fits are atomic directories. An interrupted fit is retrained from its
seed; this does not resume optimizer steps mid-fit. Published fits survive an
interruption between checkpoint publication and bank-index publication.
"""

import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import tempfile
import time
import zipfile

import torch

from amortized_gnn_tta.adaptation.checkpoints import atomic_json
from amortized_gnn_tta.adaptation_artifacts import state_sha256
from amortized_gnn_tta.data.source_bank import formal_groups, overlap_audit, json_sha256, tensor_sha256
from amortized_gnn_tta.models import make_model_spec, build_model
from amortized_gnn_tta.training import TrainingConfig, fit_source, save_checkpoint, load_checkpoint, software_versions
from prepare_arxiv_bank import load_bank_inputs
from train_adapter import code_provenance, file_hash


def hardware(device):
    if device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA is required for formal source fits')
        p = torch.cuda.get_device_properties(0)
        return dict(device='cuda', name=p.name, total_memory=p.total_memory,
                    capability=list(torch.cuda.get_device_capability(0)),
                    cuda_version=torch.version.cuda)
    return dict(device='cpu', purpose='planning or tiny fixture checks only')


def execute(a, *, allow_fixture_cpu=False):
    plan = json.loads(a.config.read_text())
    if plan.get('bank_kind') != 'formal-real-v1':
        raise ValueError('A formal bank profile is required; pilot profiles are preserved')
    if (not plan['architectures'] or len(set(plan['architectures'])) != len(plan['architectures'])
            or not set(plan['architectures']) <= {'gcn', 'gat', 'graphsage'}):
        raise ValueError('Architectures must be unique supported models')
    if plan['device'] not in {'cpu', 'cuda'}:
        raise ValueError('Use explicit cpu planning or cuda execution')
    torch.set_num_threads(plan['threads'])
    if a.fit and plan['device'] != 'cuda' and not allow_fixture_cpu:
        raise ValueError('Long laptop CPU training is prohibited; use a CUDA profile')
    # Hash all source before loading; this is the revision used by this process.
    code = code_provenance()['source_hashes']
    protocol, view, native_ids = load_bank_inputs(a, plan)
    groups = formal_groups(view, native_ids, split_seeds=plan['group_split_seeds'],
                           initialization_seeds=plan['group_initialization_seeds'])
    jobs = []
    for arch in plan['architectures']:
        for group, _ in groups:
            for seed in group['initialization_seeds']:
                spec = make_model_spec(arch, view.graph.x.shape[1], len(view.class_names),
                                       hidden_channels=plan['hidden_channels'])
                recipe = TrainingConfig(max_epochs=plan['source_epochs'],
                                        patience=plan['source_patience'], model_seed=seed,
                                        learning_rate=plan['source_learning_rate'],
                                        weight_decay=plan['source_weight_decay'])
                jobs.append(dict(id=f"{arch}-group{group['group']}-{seed}", architecture=arch,
                                 role=group['role'], group=group['group'], seed=seed,
                                 model_spec=spec, training_config=asdict(recipe)))
    identity = dict(format='formal-real-bank-v1', plan=plan, source_hashes=code,
                    software=software_versions(), hardware=hardware(plan['device']) if a.fit else None,
                    environment=protocol, groups=[g for g, _ in groups],
                    native_ids_sha256=tensor_sha256(native_ids), jobs=jobs,
                    graph_sha256=view.provenance['graph_sha256'], final_test_access=False)
    # Planning has no hardware requirement. Fitting binds the actual hardware on
    # first execution, independently from the immutable data/code/recipe identity.
    identity.pop('hardware')
    root = a.output_dir
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.bank.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = root / 'IDENTITY.json'
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != identity:
                raise ValueError('Exact bank identity mismatch; refusing resume')
        else:
            if any(p.name != '.bank.lock' for p in root.iterdir()):
                raise ValueError('Refusing to adopt a nonempty unrecognized output directory')
            atomic_json(plan_path, identity)
            atomic_json(root / 'environment_manifest.json', protocol)
            atomic_json(root / 'SUPERVISION_AUDIT.json', dict(pairs=overlap_audit(groups)))
            atomic_json(root / 'PLAN.json', dict(plan=plan, source_hashes=code, final_test_access=False))
            project = Path(__file__).resolve().parents[1]
            with zipfile.ZipFile(root / 'code_snapshot.zip', 'x') as archive:
                for name in code:
                    archive.write(project / name, name)
                archive.writestr('configs/' + a.config.name, a.config.read_bytes())
        if a.fit:
            actual = dict(hardware=hardware(plan['device']), software=software_versions())
            execution_path = root / 'EXECUTION.json'
            if execution_path.exists() and json.loads(execution_path.read_text()) != actual:
                raise ValueError('Execution hardware/dependencies changed; refusing resume')
            if not execution_path.exists():
                atomic_json(execution_path, actual)
        rows, fitted = [], 0
        models = root / 'models'
        models.mkdir(exist_ok=True)
        expected = {job['id'] for job in jobs}
        if any(p.is_dir() and p.name not in expected and not p.name.startswith('.partial-') for p in models.iterdir()):
            raise ValueError('Unexpected published model in formal bank')
        def publish_index():
            atomic_json(root / 'bank.json', dict(plan=plan, models=rows,
                        identity_sha256=json_sha256(identity),
                        status='complete' if len(rows) == len(jobs) else 'partial',
                        expected_models=len(jobs), final_test_access=False))
        previous_path = root / 'bank.json'
        previous = json.loads(previous_path.read_text())['models'] if previous_path.exists() else []
        prior = {r['id']: r for r in previous}
        if set(prior) - expected:
            raise ValueError('Unknown indexed model')
        # Verify every published checkpoint before fitting any missing job.
        for job in jobs:
            directory = models / job['id']
            if not directory.exists():
                if job['id'] in prior:
                    raise ValueError('Indexed checkpoint missing')
                continue
            row = json.loads((directory / 'record.json').read_text())
            path = directory / 'checkpoint.pt'
            model, payload = load_checkpoint(path, build_model)
            if (row['job'] != job or row['identity_sha256'] != json_sha256(identity)
                    or row['path'] != f"models/{job['id']}/checkpoint.pt"
                    or any(row[k] != job[k] for k in ('id', 'architecture', 'role', 'group', 'seed'))
                    or row['checkpoint_sha256'] != file_hash(path)
                    or row['state_sha256'] != state_sha256(model.state_dict())
                    or payload['model_spec'] != job['model_spec']
                    or payload['training_config'] != job['training_config']
                    or payload['source']['formal_group'] != groups[job['group']][0]
                    or not torch.equal(payload['source']['train_indices'], groups[job['group']][1].train_indices)
                    or not torch.equal(payload['source']['val_indices'], groups[job['group']][1].val_indices)
                    or (job['id'] in prior and prior[job['id']] != row)):
                raise ValueError('Published checkpoint identity mismatch')
            rows.append(row)
        publish_index()
        if not a.fit:
            return dict(completed=len(rows), expected=len(jobs))
        try:
            for job in jobs:
                if job['id'] in {r['id'] for r in rows}:
                    continue
                if a.max_models is not None and fitted >= a.max_models:
                    break
                print('SOURCE START', job['id'], flush=True)
                start = time.perf_counter()
                if plan['device'] == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                result = fit_source(lambda: build_model(job['model_spec']), groups[job['group']][1],
                                    TrainingConfig(**job['training_config']), device=plan['device'])
                partial = Path(tempfile.mkdtemp(prefix='.partial-', dir=models))
                path = partial / 'checkpoint.pt'
                save_checkpoint(path, result, job['model_spec'])
                restored, payload = load_checkpoint(path, build_model)
                if state_sha256(restored.state_dict()) != state_sha256(result.model.state_dict()):
                    raise ValueError('Checkpoint round-trip mismatch')
                row = dict(id=job['id'], architecture=job['architecture'], role=job['role'],
                           group=job['group'], seed=job['seed'], job=job,
                           path=f"models/{job['id']}/checkpoint.pt",
                           identity_sha256=json_sha256(identity), checkpoint_sha256=file_hash(path),
                           state_sha256=state_sha256(restored.state_dict()), selected_epoch=result.best_epoch,
                           fit_seconds=time.perf_counter()-start,
                           cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if plan['device']=='cuda' else None,
                           cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if plan['device']=='cuda' else None)
                atomic_json(partial / 'record.json', row)
                os.rename(partial, models / job['id'])
                rows.append(row)
                publish_index()
                fitted += 1
                print('SOURCE DONE', job['id'], row['fit_seconds'], flush=True)
                del result, restored
        except BaseException as error:
            atomic_json(root / 'INTERRUPTION.json', dict(error_type=type(error).__name__,
                        message=str(error), completed=len(rows), final_test_access=False))
            raise
        return dict(completed=len(rows), expected=len(jobs))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--data-root', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--fit', action='store_true')
    p.add_argument('--max-models', type=int)
    a = p.parse_args()
    if a.max_models is not None and a.max_models < 1:
        p.error('--max-models must be positive')
    print(execute(a))


if __name__ == '__main__':
    main()
