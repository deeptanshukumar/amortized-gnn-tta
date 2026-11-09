"""Private formal GCN design study. Constants are prepended by the local builder."""
from pathlib import Path
import base64,hashlib,io,json,os,shutil,subprocess,sys,threading,time,traceback,zipfile
base=Path('/kaggle/working');evidence=base/'evidence';evidence.mkdir(exist_ok=False)
started=time.time();stages=[]
def record(phase,state='running',**extra):
    p=evidence/'STATUS.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(phase=phase,state=state,elapsed_seconds=time.time()-started,**extra),indent=2));t.replace(p)
    print('AFML',phase,state,flush=True)
def command(name,args,env,timeout=5400):
    start=time.perf_counter();record(name)
    with (evidence/(name+'.log')).open('x') as log:
        process=subprocess.Popen(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,errors='replace',env=env)
        timer=threading.Timer(timeout,process.kill);timer.daemon=True;timer.start()
        try:
            for line in process.stdout:log.write(line);log.flush();print(line,end='',flush=True)
            code=process.wait()
        finally:timer.cancel()
    stages.append(dict(name=name,seconds=time.perf_counter()-start,returncode=code));(evidence/'STAGES.json').write_text(json.dumps(stages,indent=2))
    if code:raise RuntimeError(name+' failed: '+str(code))
def unpack(raw,destination):
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        manifest=json.loads(z.read('amortized-gnn-tta/colab_bundle.json'))
        if set(z.namelist())!={'amortized-gnn-tta/'+n for n in manifest['files_sha256']}|{'amortized-gnn-tta/colab_bundle.json'}:raise ValueError('Unexpected source file')
        for name,digest in manifest['files_sha256'].items():
            if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe source path')
            if hashlib.sha256(z.read('amortized-gnn-tta/'+name)).hexdigest()!=digest:raise ValueError('Source hash mismatch')
        z.extractall(destination)
    return manifest
try:
    record('verified-private-input')
    statuses=list(Path('/kaggle/input').rglob('STATUS.json'))
    if len(statuses)!=1:raise ValueError('Expected one private bank input')
    prior=statuses[0].parent
    if json.loads(statuses[0].read_text())['state']!='complete':raise ValueError('Bank job incomplete')
    for name,digest in json.loads((prior/'OUTPUT_HASHES.json').read_text()).items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe output member')
        if hashlib.sha256((prior/name).read_bytes()).hexdigest()!=digest:raise ValueError('Input hash mismatch')
    shutil.copytree(prior/'formal-bank',evidence/'formal-bank')
    bank=evidence/'formal-bank';b=json.loads((bank/'bank.json').read_text())
    if [r['checkpoint_sha256'] for r in b['models']]!=EXPECTED_CHECKPOINTS:raise ValueError('Bank checkpoint identities differ')
    raw=base64.b85decode(STUDY_ARCHIVE)
    if hashlib.sha256(raw).hexdigest()!=STUDY_SHA256:raise ValueError('Study source hash mismatch')
    (evidence/'STUDY_SOURCE.zip').write_bytes(raw)
    manifest=unpack(raw,base/'source');project=base/'source/amortized-gnn-tta';os.chdir(project)
    env=dict(os.environ,PYTHONPATH=str(project/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    command('dependencies',[sys.executable,'-m','pip','install','torch-geometric==2.8.1','ogb==1.3.6','pytest>=8,<10','psutil>=5.9,<8'],env,900)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
    torch.set_num_threads(1);sys.path.insert(0,str(project/'src'))
    from amortized_gnn_tta.adaptation.benchmark import execution_hardware
    from amortized_gnn_tta.training import software_versions
    (evidence/'HARDWARE.json').write_text(json.dumps(dict(hardware=execution_hardware('cuda'),software=software_versions(),study_source_sha256=STUDY_SHA256),indent=2))
    command('cuda-components',[sys.executable,'-m','pytest','-q','tests/test_memory_measurement.py','tests/test_conditioning_controls.py','tests/test_real_data_integration.py::test_cuda_exports_exact_generated_factors'],dict(env,AFML_CUDA_EXPORT_DIR=str(evidence/'cuda-factor-replays')),600)
    from amortized_gnn_tta.data.ogb import load_arxiv
    data=Path('/kaggle/temp/afml-ogb');data.mkdir(parents=True,exist_ok=True);load_arxiv(data,download=True)
    if (data/'archive.sha256').read_text().strip()!='49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276':raise ValueError('Arxiv archive identity changed')
    from amortized_gnn_tta.data.real_environments import ArxivEnvironments
    from amortized_gnn_tta.training import load_checkpoint
    from amortized_gnn_tta.models import build_model
    from amortized_gnn_tta.adaptation.experiments import make_descriptor
    loader=ArxivEnvironments(load_arxiv(data),json.loads((bank/'environment_manifest.json').read_text()))
    model,_=load_checkpoint(bank/b['models'][0]['path'],build_model,device='cuda');graph=loader.graph.clone().to('cuda')
    probes={};vectors={}
    for chunk in (1024,16384):
        desc,_=make_descriptor(model,'D',scope='all',chunk_size=chunk)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();vector=desc(model,graph);torch.cuda.synchronize()
        probes[str(chunk)]=dict(seconds=time.perf_counter()-t,peak_allocated_bytes=torch.cuda.max_memory_allocated());vectors[str(chunk)]=vector.cpu()
    torch.save(vectors,evidence/'DESCRIPTOR_PROBE.pt')
    torch.testing.assert_close(vectors['1024'],vectors['16384'],rtol=1e-4,atol=1e-5)
    (evidence/'DESCRIPTOR_PROBE.json').write_text(json.dumps(dict(measurements=probes,max_abs_difference=float((vectors['1024']-vectors['16384']).abs().max()),rtol=1e-4,atol=1e-5,labels_used=False,interpretation='One correctness probe; component wall times are not a matched online-method speedup'),indent=2))
    del model,graph,loader;torch.cuda.empty_cache()
    study=evidence/'controlled-study'
    common=['--bank',str(bank),'--data-root',str(data),'--output-dir',str(study)]
    command('experiments-1-2',[sys.executable,'examples/run_arxiv_experiments.py',*common,'--stop-after','2'],env,5400)
    command('conditioning-control',[sys.executable,'examples/diagnose_conditioning.py',*common],env,1800)
    baseline="from pathlib import Path; from types import SimpleNamespace; import json; from run_arxiv_experiments import Study; a=SimpleNamespace(bank=Path("+repr(str(bank))+"),data_root=Path("+repr(str(data))+"),output_dir=Path("+repr(str(study))+"),stop_after=2); s=Study(a); e=json.loads((a.output_dir/'EXPERIMENT_2.json').read_text()); s.baselines('gcn',dict(conditioning=e['selected_inputs'],update=e['selected_update']))"
    command('gcn-baselines',[sys.executable,'-c',"import sys;sys.path.insert(0,'examples');"+baseline],env,2700)
    replay=evidence/'replay_real_study.py';replay.write_text(REPLAY_CODE)
    command('independent-replay',[sys.executable,str(replay),'--bank',str(bank),'--study',str(study),'--data-root',str(data),'--device','cuda'],env,1800)
    command('report',[sys.executable,'examples/summarize_real_study.py','--bank',str(bank),'--study',str(study)],env,300)
    for name,digest in manifest['files_sha256'].items():
        if hashlib.sha256((project/name).read_bytes()).hexdigest()!=digest:raise ValueError('Source changed during study')
    record('artifact-bundle')
    with zipfile.ZipFile(evidence/'BUNDLE.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(evidence.rglob('*')):
            if p.is_file() and p.name not in {'BUNDLE.zip','STATUS.json'}:z.write(p,p.relative_to(evidence))
    hashes={str(p.relative_to(evidence)):hashlib.sha256(p.read_bytes()).hexdigest() for p in evidence.rglob('*') if p.is_file() and p.name!='STATUS.json'}
    (evidence/'OUTPUT_HASHES.json').write_text(json.dumps(hashes,indent=2))
    record('complete',state='complete',experiments=[1,2],conditioning_diagnostic=True,final_test_scored=False)
except BaseException as error:
    record('failed',state='failed',error=str(error),traceback=traceback.format_exc());raise
