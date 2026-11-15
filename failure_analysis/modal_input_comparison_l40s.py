"""Matched A/B/C/D scaled LoRA comparison. Run from the repository root."""
import os
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('AFML_REPO', str(HERE.parent)))
app = modal.App('afml-matched-inputs-l40s')
inputs = modal.Volume.from_name('afml-hypothesis-inputs', create_if_missing=True)
outputs = modal.Volume.from_name('afml-hypothesis-results', create_if_missing=True)
image = (modal.Image.debian_slim(python_version='3.13')
    .pip_install('torch==2.11.0', 'torch-geometric==2.8.1', 'numpy==2.1.3',
                 'ogb==1.3.6', 'pandas==2.3.3', 'psutil==5.9.5')
    .env({'PYTHONPATH':'/repo/src:/repo/examples:/experiment', 'OMP_NUM_THREADS':'1',
          'OPENBLAS_NUM_THREADS':'1', 'MKL_NUM_THREADS':'1'})
    .add_local_dir(str(REPO/'src'), '/repo/src')
    .add_local_dir(str(REPO/'examples'), '/repo/examples')
    .add_local_file(str(REPO/'reports/real-data-2026-11-08/design/CODE.json'), '/experiment/original_CODE.json')
    .add_local_file(str(HERE/'worker.py'), '/experiment/worker.py'))


class StopFit(Exception):
    pass


def fit_one(study, conditioning, seed, folder, commit, *, max_epochs=100, patience=10):
    """Reuse the untouched trainer, intercepting only Adam construction and epoch callbacks."""
    import json, time
    from dataclasses import asdict
    from unittest.mock import patch
    import numpy as np
    import torch
    from worker import fit_scale, attach_scale, Patience, write_json, write_csv
    from run_arxiv_experiments import create_generator
    from amortized_gnn_tta.adaptation import meta_trainer as trainer
    from amortized_gnn_tta.adaptation.checkpoints import save_generator_checkpoint

    folder=Path(folder);folder.mkdir(parents=True, exist_ok=False)
    models=study.sources('gcn','meta_train'); validation=study.sources('gcn','validation')
    if (len(models),len(validation),len(study.validation)) != (18,6,7):
        raise ValueError('Expected the original 18/6 GCN instances and seven validation graphs')
    # Load source instances BEFORE seeding, consistently for every configuration.
    torch.manual_seed(seed)
    descriptor,generator,schema,scope=create_generator(next(iter(models.values())),conditioning,'lora','hypernetwork',study.plan)
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        normalizer=fit_scale(study,descriptor,models)
    attach_scale(generator,normalizer['scale'])
    torch.save(normalizer,folder/'scaling.pt')
    lr0=float(study.plan['generator_lr']); lr_min=1e-5
    if lr0 <= lr_min: raise ValueError('Expected original initial learning rate above 1e-5')
    recipe=dict(conditioning=conditioning,update='lora',seed=seed,max_epochs=max_epochs,patience=patience,
        scaling=normalizer['definition'],lr_initial=lr0,lr_min=lr_min,
        scheduler='CosineAnnealingLR',T_max=max_epochs,scheduler_step='after each completed training epoch',
        checkpoint_selection='best mean validation accuracy including epoch zero; strict improvement',
        patience_monitor='best trained-epoch mean validation accuracy; excludes epoch zero',
        loss='original CE + original delta_l2 * sum(dense_delta**2)',
        initialization='all source models preloaded, then torch.manual_seed(seed) for each input version',
        final_test_scored=False)
    write_json(folder/'RECIPE.json',recipe)
    captured={}; real_adam=torch.optim.Adam
    def construct_adam(*args,**kwargs):
        if captured: raise RuntimeError('Unexpected second optimizer inside direct trainer')
        opt=real_adam(*args,**kwargs)
        captured['optimizer']=opt
        captured['scheduler']=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max_epochs,eta_min=lr_min)
        return opt
    history=[]; cases=[]; state_best=None; best_error=float('inf'); selected_epoch=0; eval_epoch=0
    monitor=Patience(patience); original_evaluate=trainer.evaluate_adaptation_pair
    def evaluate(*args,**kwargs):
        record=original_evaluate(*args,**kwargs)
        cases.append(dict(epoch=eval_epoch,**asdict(record)))
        return record
    started=time.perf_counter()
    def checkpoint(state, records):
        nonlocal history,state_best,best_error,selected_epoch,eval_epoch
        epoch=len(records); opt=captured['optimizer'];scheduler=captured['scheduler']
        current=[r for r in cases if r['epoch']==epoch]
        if len(current)!=42: raise RuntimeError('Incomplete validation pass')
        error=1-float(np.mean([r['adapted_accuracy'] for r in current]))
        if error < best_error:best_error=error;selected_epoch=epoch
        state_best={k:v.detach().cpu().clone() for k,v in state.items()}
        if epoch:
            row=dict(records[-1]);row['lr_used']=opt.param_groups[0]['lr']
            scheduler.step()
            row['lr_next']=opt.param_groups[0]['lr'];history.append(row)
        torch.save(dict(epoch=epoch,state_dict={k:v.detach().cpu().clone() for k,v in generator.state_dict().items()},
                        scaling_file='scaling.pt',inference_only=True),folder/f'epoch_{epoch:03d}.pt')
        torch.save(dict(epoch=selected_epoch,state_dict=state_best),folder/'best_state.pt')
        # Latest optimizer/scheduler/RNG state is saved for forensic analysis.
        # Automatic mid-epoch/mid-fit resumption is deliberately not implemented.
        torch.save(dict(epoch=epoch,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),
                        cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),folder/'latest_training_state.pt')
        write_csv(folder/'history.csv',history);write_csv(folder/'validation_cases.csv',cases)
        write_json(folder/'STATUS.json',dict(state='training',epoch=epoch,selected_epoch=selected_epoch,
                                           validation_gain_pp=100*float(np.mean([r['accuracy_gain'] for r in current]))))
        commit();eval_epoch=epoch+1
        if epoch:
            print(f'{conditioning} seed={seed} epoch={epoch} lr_used={history[-1]["lr_used"]:.8g} '
                  f'lr_next={history[-1]["lr_next"]:.8g}',flush=True)
            if monitor.update(error):raise StopFit()
    stopped=False;trainer.evaluate_adaptation_pair=evaluate
    try:
        # Narrow runtime adapter: original Adam, with identical original kwargs.
        # The original training source and its optimization loop remain unchanged.
        with patch.object(trainer.torch.optim,'Adam',side_effect=construct_adam):
            trainer.train_adaptation_generator(models=models,manifest=study.manifest,load_env_fn=study.loader,
                descriptor=descriptor,generator=generator,val_models=validation,epochs=max_epochs,
                lr=lr0,delta_l2_reg=study.plan['delta_l2'],target_scope=scope,
                device=study.device,checkpoint_callback=checkpoint)
    except StopFit:stopped=True
    except BaseException as error:
        write_json(folder/'FAILURE.json',dict(type=type(error).__name__,message=str(error)));commit();raise
    finally:trainer.evaluate_adaptation_pair=original_evaluate
    elapsed=time.perf_counter()-started
    if (folder/'best_generator.pt').exists():raise FileExistsError(folder/'best_generator.pt')
    save_generator_checkpoint(folder/'best_generator.pt',generator,descriptor,schema,target_scope=scope,
        provenance=dict(**recipe,selected_epoch=selected_epoch,load_instructions='Use worker.load_for_analysis to restore RMS scaling'),
        history=history,state_dict=state_best,overwrite=True)  # Atomic replace; no Modal hard links.
    generator.load_state_dict(state_best)
    records=study.evaluate('gcn',generator,descriptor,scope,'hypernetwork',folder)
    write_json(folder/'SELECTED_VALIDATION.json',records)
    chosen=[r for r in cases if r['epoch']==selected_epoch]
    result=dict(arm='scaled_cosine',conditioning=conditioning,update='lora',seed=seed,
        epochs_completed=len(history),selected_epoch=selected_epoch,
        selected_validation_accuracy=1-best_error,
        selected_gain_pp=100*float(np.mean([r['accuracy_gain'] for r in chosen])),
        early_stopped=stopped,optimizer_steps=126*len(history),
        training_and_epoch_export_seconds=elapsed,final_test_scored=False)
    write_json(folder/'RESULT.json',result);write_json(folder/'STATUS.json',dict(state='complete',**result));commit()
    return result


@app.function(image=image,gpu='L40S',cpu=4,memory=32768,timeout=24*60*60,
              volumes={'/inputs':inputs,'/results':outputs},retries=0)
def train_seed(run_id: str, seed: int):
    import json,sys
    import torch
    from types import SimpleNamespace
    from worker import prepare_runtime_source,write_json,write_csv
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ValueError('Use only letters, digits, hyphens and underscores in run-id')
    if not torch.cuda.is_available() or 'L40S' not in torch.cuda.get_device_name(0):
        raise RuntimeError('An actual NVIDIA L40S is required')
    repo=prepare_runtime_source('/repo','/tmp/afml-input-comparison-source')
    sys.path[:0]=[str(repo/'src'),str(repo/'examples')]
    from train_adapter import code_provenance,file_hash
    from run_arxiv_experiments import Study
    from amortized_gnn_tta.training import software_versions
    from amortized_gnn_tta.adaptation.benchmark import execution_hardware
    expected=json.loads(Path('/experiment/original_CODE.json').read_text())['code']['source_hashes']
    if code_provenance()['source_hashes']!=expected:raise ValueError('Original source hashes changed')
    root=Path('/results')/run_id/f'seed_{seed}'
    root.mkdir(parents=True,exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
    torch.set_float32_matmul_precision('highest')
    study=Study(SimpleNamespace(bank=Path('/inputs/bank'),data_root=Path('/inputs/arxiv'),
                               output_dir=root/'study_context',stop_after=2))
    if study.plan.get('bank_kind')!='formal-real-v1' or study.plan.get('device')!='cuda':
        raise ValueError('Existing formal CUDA source bank required')
    write_json(root/'PROVENANCE.json',dict(code=code_provenance(),software=software_versions(),
        hardware=execution_hardware('cuda'),script_sha256=file_hash(Path(__file__)),
        worker_sha256=file_hash(Path('/experiment/worker.py')),final_test_scored=False))
    results=[]
    for conditioning in 'ABCD':
        results.append(fit_one(study,conditioning,seed,root/f'{conditioning}-lora',outputs.commit))
        write_csv(root/'summary.csv',results);outputs.commit()
    write_json(root/'COMPLETE.json',dict(results=results,final_test_scored=False));outputs.commit()
    return results


@app.local_entrypoint()
def main(run_id: str='',seeds: str='202801,202802,202803'):
    if not run_id:raise ValueError('Supply a unique --run-id')
    seed_list=[int(s.strip()) for s in seeds.split(',')]
    if len(seed_list)!=len(set(seed_list)) or any(not 0<=s<2**32 for s in seed_list):
        raise ValueError('Seeds must be distinct integers in [0,2**32)')
    for seed in seed_list:print(train_seed.remote(run_id,seed))  # Sequential, one L40S at a time.
