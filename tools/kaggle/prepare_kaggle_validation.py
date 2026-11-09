"""Build a private self-contained Kaggle validation script from an exact code ZIP.

No credentials are read. Kaggle authentication is handled only by its official CLI.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-zip',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--template',type=Path,required=True)
    p.add_argument('--slug',required=True)
    a=p.parse_args()
    raw=a.source_zip.read_bytes()
    with zipfile.ZipFile(a.source_zip) as z:
        manifest=json.loads(z.read('amortized-gnn-tta/colab_bundle.json'))
        expected={'amortized-gnn-tta/'+s for s in manifest['files_sha256']} | {'amortized-gnn-tta/colab_bundle.json'}
        if set(z.namelist()) != expected or manifest['bank'] is not None:
            raise ValueError('Expected exact code-only archive allowlist')
        for name, digest in manifest['files_sha256'].items():
            if hashlib.sha256(z.read('amortized-gnn-tta/'+name)).hexdigest()!=digest:
                raise ValueError('Snapshot hash mismatch')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    template=json.loads(a.template.read_text()); owner=template['id'].split('/')[0]
    metadata=dict(template,id=owner+'/'+a.slug,title=a.slug.replace('-',' '),code_file='validation.py',
                  language='python',kernel_type='script',is_private=True,enable_gpu=True,
                  enable_tpu=False,enable_internet=True,machine_shape='NvidiaTeslaT4',
                  dataset_sources=[],competition_sources=[],kernel_sources=[],model_sources=[])
    header='ARCHIVE_B85 = '+repr(base64.b85encode(raw).decode())+'\nARCHIVE_SHA256 = '+repr(hashlib.sha256(raw).hexdigest())+'\n'
    driver=r'''
from pathlib import Path
import base64, hashlib, io, json, os, platform, subprocess, sys, time, traceback, zipfile
base=Path('/kaggle/working')
evidence=base/'evidence'; evidence.mkdir(exist_ok=False)
started=time.time()
def record(phase, state='running', **more):
    payload=dict(phase=phase,state=state,elapsed_seconds=time.time()-started,**more)
    path=evidence/'STATUS.json'; temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload,indent=2)); temp.replace(path)
    print('AFML',phase,state,flush=True)
def command(name, args, *, env=None, timeout=1800):
    print('START',name,flush=True)
    with (evidence/(name+'.log')).open('x') as log:
        result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,env=env,timeout=timeout)
    if result.returncode:
        print((evidence/(name+'.log')).read_text()[-12000:],flush=True)
        raise RuntimeError(name+' failed: '+str(result.returncode))
    print('DONE',name,flush=True)
try:
    record('immutable-source')
    archive=base64.b85decode(ARCHIVE_B85)
    if hashlib.sha256(archive).hexdigest()!=ARCHIVE_SHA256:
        raise ValueError('Source archive hash mismatch')
    (evidence/'SOURCE.zip').write_bytes(archive)
    with zipfile.ZipFile(io.BytesIO(archive)) as z:
        manifest=json.loads(z.read('amortized-gnn-tta/colab_bundle.json'))
        expected={'amortized-gnn-tta/'+s for s in manifest['files_sha256']} | {'amortized-gnn-tta/colab_bundle.json'}
        if set(z.namelist())!=expected:
            raise ValueError('Unexpected archive member')
        for name, digest in manifest['files_sha256'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:
                raise ValueError('Unsafe archive member')
            if hashlib.sha256(z.read('amortized-gnn-tta/'+name)).hexdigest()!=digest:
                raise ValueError('Source file hash mismatch')
        z.extractall(base/'source')
    project=base/'source/amortized-gnn-tta'; os.chdir(project)
    record('dependencies')
    command('install',[sys.executable,'-m','pip','install','torch-geometric==2.8.1','ogb==1.3.6','pytest>=8,<10','psutil>=5.9,<8'],timeout=900)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback')
    torch.set_num_threads(1)
    hardware=dict(python=sys.version,platform=platform.platform(),torch=str(torch.__version__),
                  cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                  gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  source_zip_sha256=ARCHIVE_SHA256)
    (evidence/'HARDWARE.json').write_text(json.dumps(hardware,indent=2))
    command('dependencies',[sys.executable,'-m','pip','freeze'])
    env=dict(os.environ,PYTHONPATH=str(project/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    record('clean-process-regression')
    command('regression',[sys.executable,'-m','pytest','-q'],env=dict(env,CUDA_VISIBLE_DEVICES=''),timeout=1800)
    record('cuda-component-checks')
    command('cuda-adaptation',[sys.executable,'examples/check_adaptation_device.py','--device','cuda','--output-dir',str(evidence/'cuda-adaptation')],env=env,timeout=600)
    command('cuda-factor-replay',[sys.executable,'-m','pytest','-q','tests/test_real_data_integration.py::test_cuda_exports_exact_generated_factors'],
            env=dict(env,AFML_CUDA_EXPORT_DIR=str(evidence/'factor-replays')),timeout=600)
    sys.path.insert(0,str(project/'src'))
    from amortized_gnn_tta.data.ogb import load_arxiv
    record('official-arxiv')
    data=Path('/kaggle/temp/afml-ogb'); data.mkdir(parents=True,exist_ok=True)
    load_arxiv(data,download=True)
    if (data/'archive.sha256').read_text().strip()!='49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276':
        raise ValueError('Official Arxiv download differs from verified local archive')
    command('official-data',[sys.executable,'-m','pytest','-q','tests/test_real_data_integration.py::test_official_real_graph_protocols_and_all_architecture_forward'],env=dict(env,AFML_OGB_ROOT=str(data)),timeout=600)
    record('one-formal-source-fit')
    bank=evidence/'formal-bank'
    config=project/'configs/arxiv_formal_v1.json'
    common=[sys.executable,'examples/prepare_formal_bank.py','--config',str(config),'--data-root',str(data),'--output-dir',str(bank)]
    command('one-source-fit',common+['--fit','--max-models','1'],env=env,timeout=1800)
    before=json.loads((bank/'bank.json').read_text())
    if len(before['models'])!=1 or before['status']=='complete':
        raise ValueError('One-fit preflight must remain a partial bank')
    record('verify-resume-without-fitting')
    command('resume-verification',common,env=env,timeout=600)
    after=json.loads((bank/'bank.json').read_text())
    if after['models']!=before['models']:
        raise ValueError('Resume changed published source checkpoint')
    # Recheck original captured bytes after all tests/fitting.
    for name, digest in manifest['files_sha256'].items():
        if hashlib.sha256((project/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Source changed during job')
    hashes={str(path.relative_to(evidence)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in evidence.rglob('*') if path.is_file() and path.name!='STATUS.json'}
    (evidence/'OUTPUT_HASHES.json').write_text(json.dumps(hashes,indent=2))
    record('complete',state='complete',completed_source_fits=1,expected_source_fits=90,
           final_test_scored=False,remote_versioned_resume_verified=False)
except BaseException as error:
    record('failed',state='failed',error=str(error),traceback=traceback.format_exc())
    raise
'''
    ast.parse(header+driver)
    (a.output_dir/'validation.py').write_text(header+driver)
    (a.output_dir/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (a.output_dir/'LOCAL_MANIFEST.json').write_text(json.dumps(dict(source_zip_sha256=hashlib.sha256(raw).hexdigest(),
        source_files=len(manifest['files_sha256']),script_sha256=hashlib.sha256((header+driver).encode()).hexdigest(),
        is_private=True,timeout_seconds=7200),indent=2)+'\n')
    print(json.dumps(dict(kernel=metadata['id'],is_private=True,source_files=len(manifest['files_sha256']))))

if __name__=='__main__': main()
