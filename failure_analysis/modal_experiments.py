"""modal run --detach failure_analysis/modal_experiments.py --arm all --run-id YOUR_UNIQUE_ID"""
import os
from pathlib import Path
import modal

HERE = Path(__file__).resolve().parent
REPO = Path(os.environ.get('AFML_REPO', str(HERE.parent)))
app = modal.App('afml-arxiv-hypotheses')
inputs = modal.Volume.from_name('afml-hypothesis-inputs', create_if_missing=True)
outputs = modal.Volume.from_name('afml-hypothesis-results', create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version='3.13')
    .pip_install('torch==2.11.0', 'torch-geometric==2.8.1', 'numpy==2.1.3',
                 'ogb==1.3.6', 'pandas==2.3.3', 'psutil==5.9.5')
    .env({'PYTHONPATH': '/repo/src:/repo/examples:/experiment',
          'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'})
    .add_local_dir(str(REPO / 'src'), '/repo/src')
    .add_local_dir(str(REPO / 'examples'), '/repo/examples')
    .add_local_file(str(REPO / 'reports/real-data-2026-11-08/design/CODE.json'), '/experiment/original_CODE.json')
    .add_local_file(str(HERE / 'worker.py'), '/experiment/worker.py')
)


@app.function(image=image, gpu='A10', cpu=4, memory=32768, timeout=24*60*60,
              volumes={'/inputs': inputs, '/results': outputs}, retries=0)
def train(arm: str, run_id: str, configurations: str):
    from worker import run
    if not run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_' for c in run_id):
        raise ValueError('run-id must contain only letters, digits, hyphens or underscores')
    # Keep this byte-identical copy for the CONTAINER lifetime. Warm containers
    # retain imported modules whose __file__ points into this directory.
    # Deleting a per-call copy leaves provenance scans pointing at missing files.
    from worker import prepare_runtime_source
    runtime_repo = prepare_runtime_source('/repo', '/tmp/afml-hypothesis-source')
    return run(runtime_repo, '/inputs/bank', '/inputs/arxiv', f'/results/{run_id}',
               arm, outputs.commit, configurations, '/experiment/original_CODE.json')


@app.local_entrypoint()
def main(arm: str = 'all', run_id: str = '', configurations: str = 'all'):
    if not run_id:
        raise ValueError('Supply a unique --run-id; existing runs are never overwritten')
    arms = (['control', 'epochs', 'scaling'] if arm == 'all' else
            ['epochs', 'scaling'] if arm == 'remaining' else [arm])
    if any(a not in {'control', 'epochs', 'scaling'} for a in arms):
        raise ValueError('arm must be all, remaining, control, epochs or scaling')
    # Sequential remote calls: one A10 at a time, no duplicate concurrent matrix.
    for name in arms:
        print(train.remote(name, run_id, configurations))
