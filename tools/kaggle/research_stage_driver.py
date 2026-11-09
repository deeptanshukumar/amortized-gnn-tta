"""Next-stage worker. Common immutable-source helpers are prepended by builder."""
def file_digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def verify_evidence(folder):
    if json.loads((folder/'STATUS.json').read_text())['state']!='complete':raise ValueError('Incomplete upstream stage')
    for name,digest in json.loads((folder/'OUTPUT_HASHES.json').read_text()).items():
        if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe upstream path')
        if file_digest(folder/name)!=digest:raise ValueError('Upstream artifact hash mismatch: '+name)

def copy_study(source,destination,thin=False):
    destination.mkdir()
    for path in source.rglob('*'):
        if not path.is_file():continue
        rel=path.relative_to(source)
        if thin and len(rel.parts)>1 and path.name not in {'generator.pt','RESULT.json'}:continue
        target=destination/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    if thin:(destination/'PRIOR_ARTIFACT_REFERENCE.json').write_text(json.dumps(dict(private_source=STUDY_INPUT,validation_artifacts_preserved_in_prior_version=True),indent=2))

try:
    record('verify-upstream')
    upstream=[p.parent for p in Path('/kaggle/input').rglob('STATUS.json')]
    for folder in upstream:verify_evidence(folder)
    bank_sources=[];study_sources=[]
    for folder in upstream:
        path=folder/'formal-bank/bank.json'
        if path.exists():
            b=json.loads(path.read_text())
            if b['plan'].get('dataset','ogbn-arxiv')==DATASET and len(b['models'])==90:bank_sources.append(folder)
        if (folder/'controlled-study').exists():study_sources.append(folder)
    if not bank_sources:raise ValueError('Required complete source bank is unavailable')
    bank_source=bank_sources[0];shutil.copytree(bank_source/'formal-bank',evidence/'formal-bank');bank=evidence/'formal-bank'
    if not study_sources:raise ValueError('Required frozen study input missing')
    prior=study_sources[0]/'controlled-study'
    raw=base64.b85decode(STUDY_ARCHIVE)
    if hashlib.sha256(raw).hexdigest()!=STUDY_SHA256:raise ValueError('Study snapshot mismatch')
    (evidence/'STUDY_SOURCE.zip').write_bytes(raw);manifest=unpack(raw,base/'source')
    project=base/'source/amortized-gnn-tta';os.chdir(project)
    env=dict(os.environ,PYTHONPATH=str(project/'src'),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    command('dependencies',[sys.executable,'-m','pip','install','torch-geometric==2.8.1','ogb==1.3.6','pytest>=8,<10','psutil>=5.9,<8'],env,900)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Actual CUDA required')
    torch.set_num_threads(1);sys.path.insert(0,str(project/'src'))
    from amortized_gnn_tta.adaptation.benchmark import execution_hardware
    from amortized_gnn_tta.training import software_versions
    (evidence/'HARDWARE.json').write_text(json.dumps(dict(hardware=execution_hardware('cuda'),software=software_versions(),study_source_sha256=STUDY_SHA256),indent=2))
    if DATASET=='pair-align-mag-country':
        from amortized_gnn_tta.data.natural import load_country
        data=Path('/kaggle/temp/afml-mag');data.mkdir(parents=True,exist_ok=True)
        for country in ('US','CN','DE','FR','JP','RU'):load_country(data,country,download=True)
    else:
        from amortized_gnn_tta.data.ogb import load_arxiv
        data=Path('/kaggle/temp/afml-ogb');data.mkdir(parents=True,exist_ok=True);load_arxiv(data,download=True)
        if (data/'archive.sha256').read_text().strip()!='49f85c801589ecdcc52cfaca99693aaea7b8af16a9ac3f41dd85a5f3193fe276':raise ValueError('Arxiv artifact identity mismatch')
    study=evidence/'controlled-study'
    if MODE in {'extend','final','diagnostic'}:copy_study(prior,study,thin=MODE in {'final','diagnostic'})
    else:study.mkdir()
    common=['--bank',str(bank),'--data-root',str(data),'--output-dir',str(study)]
    if MODE=='extend':
        command('experiment-3',[sys.executable,'examples/run_arxiv_experiments.py',*common],env,STAGE_TIMEOUT)
    elif MODE=='natural':
        design=prior/'FROZEN_DESIGN.json'
        if not design.exists():raise ValueError('Controlled design is not frozen')
        command('experiment-5-validation',[sys.executable,'examples/run_arxiv_experiments.py',*common,'--natural-design',str(design)],env,STAGE_TIMEOUT)
    elif MODE=='final':
        runner=evidence/'evaluate_final_shard.py';runner.write_text(FINAL_RUNNER)
        command('final-'+ARCHITECTURE,[sys.executable,str(runner),*common,'--architecture',ARCHITECTURE,'--final-evaluation','--measure-memory'],dict(env,PYTHONPATH=str(project/'src')+':'+str(project/'examples')),STAGE_TIMEOUT)
        plan=json.loads((study/'FINAL/EVALUATION_PLAN.json').read_text())
        if plan['architecture_shard']!=ARCHITECTURE:raise ValueError('Final shard identity mismatch')
    elif MODE=='diagnostic':
        runner=evidence/'diagnose_real_audit.py';runner.write_text(DIAGNOSTIC_RUNNER)
        command('experiment-6-diagnostic',[sys.executable,str(runner),*common],dict(env,PYTHONPATH=str(project/'src')+':'+str(project/'examples')),STAGE_TIMEOUT)
    else:raise ValueError('Unknown stage mode')
    if MODE in {'extend','natural'}:
        replay=evidence/'replay_real_study.py';replay.write_text(REPLAY_CODE)
        command('independent-replay',[sys.executable,str(replay),'--bank',str(bank),'--study',str(study),'--data-root',str(data),'--device','cuda'],env,1800)
    command('report',[sys.executable,'examples/summarize_real_study.py','--bank',str(bank),'--study',str(study)],env,300)
    for name,digest in manifest['files_sha256'].items():
        if file_digest(project/name)!=digest:raise ValueError('Immutable source changed')
    record('archive-evidence')
    with zipfile.ZipFile(evidence/'BUNDLE.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(evidence.rglob('*')):
            if path.is_file() and path.name not in {'BUNDLE.zip','STATUS.json'}:z.write(path,path.relative_to(evidence))
    hashes={str(p.relative_to(evidence)):file_digest(p) for p in evidence.rglob('*') if p.is_file() and p.name!='STATUS.json'}
    (evidence/'OUTPUT_HASHES.json').write_text(json.dumps(hashes,indent=2))
    record('complete',state='complete',stage=MODE,dataset=DATASET,architecture_shard=ARCHITECTURE if MODE=='final' else None,final_test_scored=MODE=='final')
except BaseException as error:
    record('failed',state='failed',stage=MODE,error=str(error),traceback=traceback.format_exc());raise
