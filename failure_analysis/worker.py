"""Read-only use of the original AFML trainer; no replacement training loop."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import torch


def prepare_runtime_source(source, destination):
    """Stable source path across warm invocations, with ZIP-compatible mtimes."""
    import shutil
    source, destination = Path(source), Path(destination)
    if not destination.exists():
        shutil.copytree(source, destination, copy_function=shutil.copyfile)
        for source_file in destination.rglob('*.py'):
            os.utime(source_file, (315619200, 315619200))
    return destination


def write_json(path, value):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    temporary = Path(str(path) + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class EarlyStop(Exception):
    pass


class Patience:
    """Monitor trained epochs, independently of identity checkpoint selection."""
    def __init__(self, limit):
        self.limit, self.best, self.stale = limit, float('inf'), 0

    def update(self, error):
        if error < self.best:
            self.best, self.stale = error, 0
        else:
            self.stale += 1
        return self.limit is not None and self.stale >= self.limit


def attach_scale(generator, scale):
    """Scale only conditioning at the core MLP entrance, for train and inference."""
    core = getattr(generator, 'core', generator)

    def scale_input(module, args):
        return (args[0] / scale.to(args[0]), *args[1:])

    return core.register_forward_pre_hook(scale_input)


def load_for_analysis(folder, model, device='cpu'):
    """Canonical checkpoint loader plus the REQUIRED saved input scaling hook."""
    from amortized_gnn_tta.adaptation.checkpoints import load_generator_checkpoint
    folder = Path(folder)
    generator, descriptor, payload = load_generator_checkpoint(
        folder / 'best_generator.pt', model=model, device=device)
    normalizer = torch.load(folder / 'scaling.pt', weights_only=True, map_location='cpu')
    if normalizer['enabled']:
        attach_scale(generator, normalizer['scale'])
    return generator, descriptor, payload


@torch.no_grad()
def fit_scale(study, descriptor, models):
    """Per-coordinate RMS across ALL 126 training pairs; never validation labels."""
    from amortized_gnn_tta.adaptation._validation import source_state
    vectors = []
    for env in study.manifest['environments']:
        if env['role'] != 'meta_train':
            continue
        graph = study.loader(study.manifest, env['id']).unlabeled_input(env['id']).to(study.device)
        for model in models.values():
            with source_state(model):
                vectors.append(descriptor(model, graph).detach().cpu())
    raw = torch.stack(vectors)
    scale = raw.double().square().mean(0).sqrt().clamp_min(1e-6).float()
    return dict(enabled=True, scale=scale, raw_training_descriptors=raw,
                definition='sqrt(mean(training_descriptor**2)), coordinatewise, floor=1e-6; no centering',
                training_pairs=len(raw), labels_used=False)


def run(repo, bank, data_root, output_root, arm, commit=lambda: None,
        configurations='all', expected_source=None):
    repo, bank, data_root, output_root = map(Path, (repo, bank, data_root, output_root))
    sys.path[:0] = [str(repo / 'src'), str(repo / 'examples')]
    from run_arxiv_experiments import Study, create_generator
    from train_adapter import code_provenance, file_hash
    from amortized_gnn_tta.adaptation import meta_trainer as trainer
    from amortized_gnn_tta.adaptation.checkpoints import save_generator_checkpoint
    from amortized_gnn_tta.training import software_versions
    from amortized_gnn_tta.adaptation.benchmark import execution_hardware
    from dataclasses import asdict
    import numpy as np

    if arm not in {'control', 'epochs', 'scaling'}:
        raise ValueError(arm)
    if not torch.cuda.is_available() or 'A10' not in torch.cuda.get_device_name(0):
        raise RuntimeError('An actual NVIDIA A10 is required')
    if expected_source:
        expected = json.loads(Path(expected_source).read_text())['code']['source_hashes']
        actual = code_provenance()['source_hashes']
        if actual != expected:
            missing = sorted(set(expected) - set(actual))
            added = sorted(set(actual) - set(expected))
            changed = sorted(k for k in set(actual) & set(expected) if actual[k] != expected[k])
            raise ValueError(f'Source differs from the audited original experiment: '
                             f'missing={missing}; added={added}; changed={changed}')
    root = output_root / arm
    root.mkdir(parents=True, exist_ok=False)  # Never overwrite a previous arm.
    os.environ['OMP_NUM_THREADS'] = '1'
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('highest')
    study = Study(SimpleNamespace(bank=bank, data_root=data_root,
                                output_dir=root / 'study_context', stop_after=2))
    if study.plan.get('bank_kind') != 'formal-real-v1' or study.plan.get('device') != 'cuda':
        raise ValueError('Use the existing formal CUDA bank, not a new pilot bank')
    epochs, patience = (100, 10) if arm == 'epochs' else (5, None)
    configs = [('A', 'lora'), ('B', 'lora'), ('C', 'lora'), ('D', 'lora'),
               ('A', 'head'), ('D', 'head')]
    if configurations != 'all':
        requested = set(configurations.split(','))
        valid = {f'{c}-{u}' for c, u in configs}
        if not requested <= valid:
            raise ValueError(f'Choose configurations from {sorted(valid)}')
        configs = [(c, u) for c, u in configs if f'{c}-{u}' in requested]
    seed = study.plan['generator_seeds'][0]
    recipe = dict(arm=arm, max_epochs=epochs, patience=patience,
                  stopping_monitor='best trained-epoch mean validation accuracy; epoch zero excluded from patience',
                  checkpoint_monitor='best mean validation accuracy INCLUDING epoch zero; strict improvement',
                  configurations=configs, generator_seed=seed,
                  original_plan=study.plan, code=code_provenance(), software=software_versions(),
                  hardware=execution_hardware('cuda'), final_test_scored=False,
                  runner_sha256=file_hash(Path(__file__)),
                  scaling='training-pair coordinate RMS, no centering' if arm == 'scaling' else 'none')
    write_json(root / 'RECIPE.json', recipe)
    summaries = []
    for conditioning, update in configs:
        folder = root / f'{conditioning}-{update}'
        folder.mkdir()
        # Preserve original Study.fit order: seed, source-model loading, generator,
        # then validation-model loading. Model loading can consume RNG state.
        torch.manual_seed(seed)
        model = next(iter(study.sources('gcn', 'meta_train').values()))
        descriptor, generator, schema, scope = create_generator(
            model, conditioning, update, 'hypernetwork', study.plan)
        models = study.sources('gcn', 'meta_train')
        val_models = study.sources('gcn', 'validation')
        if (len(models), len(val_models), len(study.validation)) != (18, 6, 7):
            raise ValueError('Expected the original 18 training / 6 validation models and 7 validation graphs')
        normalizer = dict(enabled=False, scale=torch.ones(descriptor.output_dim))
        if arm == 'scaling':
            # Extra statistics pass must not change training RNG or model modes.
            numpy_state, python_state = np.random.get_state(), random.getstate()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                normalizer = fit_scale(study, descriptor, models)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            attach_scale(generator, normalizer['scale'])
        torch.save(normalizer, folder / 'scaling.pt')
        patience_monitor = Patience(patience)
        history, cases = [], []
        best_epoch, best_error = 0, float('inf')
        best_state = None
        original_evaluate = trainer.evaluate_adaptation_pair
        current_epoch = 0

        def capture(*args, **kwargs):
            record = original_evaluate(*args, **kwargs)
            cases.append(dict(epoch=current_epoch, **asdict(record)))
            return record

        started = time.perf_counter()

        def checkpoint(state, records):
            nonlocal history, best_epoch, best_error, best_state, current_epoch
            epoch = len(records)
            history = [dict(r) for r in records]
            epoch_cases = [r for r in cases if r['epoch'] == epoch]
            expected = len(val_models) * len(study.validation)
            if len(epoch_cases) != expected:
                raise ValueError(f'Expected {expected} validation records, got {len(epoch_cases)}')
            # Match summarize_records' NumPy aggregation exactly.
            error = 1 - float(np.mean([r['adapted_accuracy'] for r in epoch_cases]))
            if error < best_error:
                best_epoch, best_error = epoch, error
            best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            # The callback's state is BEST; generator.state_dict() is THIS epoch.
            torch.save(dict(epoch=epoch, state_dict={k: v.detach().cpu().clone()
                       for k, v in generator.state_dict().items()},
                       scaling_file='scaling.pt', inference_only=True),
                       folder / f'epoch_{epoch:03d}.pt')
            torch.save(dict(epoch=best_epoch, state_dict=best_state), folder / 'best_state.pt')
            write_csv(folder / 'history.csv', history)
            write_csv(folder / 'validation_cases.csv', cases)
            write_json(folder / 'STATUS.json', dict(state='training', epoch=epoch,
                       selected_epoch=best_epoch, best_validation_accuracy=1-best_error))
            commit()  # Save every epoch to the persistent Modal volume.
            current_epoch = epoch + 1
            if epoch and patience_monitor.update(error):
                raise EarlyStop()

        trainer.evaluate_adaptation_pair = capture
        stopped = False
        try:
            trainer.train_adaptation_generator(
                models=models, manifest=study.manifest, load_env_fn=study.loader,
                descriptor=descriptor, generator=generator, val_models=val_models,
                epochs=epochs, lr=study.plan['generator_lr'],
                delta_l2_reg=study.plan['delta_l2'], target_scope=scope,
                device=study.device, checkpoint_callback=checkpoint)
        except EarlyStop:
            stopped = True
        except BaseException as error:
            write_json(folder / 'FAILURE.json', dict(type=type(error).__name__, message=str(error)))
            commit()
            raise
        finally:
            trainer.evaluate_adaptation_pair = original_evaluate
        elapsed = time.perf_counter() - started
        # Modal's results volume rejects hard links. The original writer's
        # overwrite=True branch uses atomic replace instead. This configuration
        # folder was exclusively created by this invocation, so no concurrent
        # writer is allowed; still refuse an existing selected checkpoint.
        if (folder / 'best_generator.pt').exists():
            raise FileExistsError(folder / 'best_generator.pt')
        save_generator_checkpoint(folder / 'best_generator.pt', generator, descriptor,
            schema, target_scope=scope, provenance=dict(**recipe, conditioning=conditioning,
            update=update, selected_epoch=best_epoch, scaling_file='scaling.pt',
            load_instructions='Use worker.load_for_analysis; canonical loader alone omits RMS scaling'),
            history=history, state_dict=best_state, overwrite=True)
        row = dict(arm=arm, conditioning=conditioning, update=update,
                   epochs_completed=len(history), selected_epoch=best_epoch,
                   selected_validation_accuracy=1-best_error,
                   selected_gain_pp=100*((1-best_error)-sum(r['baseline_accuracy'] for r in cases[:42])/42),
                   early_stopped=stopped, optimizer_steps=126*len(history),
                   training_and_epoch_export_seconds=elapsed,
                   final_test_scored=False)
        # Save selected update factors/dense deltas/logits through the ORIGINAL evaluator.
        generator.load_state_dict(best_state)
        records = study.evaluate('gcn', generator, descriptor, scope, 'hypernetwork', folder)
        write_json(folder / 'SELECTED_VALIDATION.json', records)
        write_json(folder / 'RESULT.json', row)
        write_json(folder / 'STATUS.json', dict(state='complete', **row))
        summaries.append(row)
        write_csv(root / 'summary.csv', summaries)
        commit()
    write_json(root / 'COMPLETE.json', dict(results=summaries, final_test_scored=False))
    commit()
    return summaries
