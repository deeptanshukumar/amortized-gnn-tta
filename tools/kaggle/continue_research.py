"""Durable sequential execution of already-authorized experiments, with strict gates.

No methodological choices or Git operations. Read errors retry; remote submissions
never retry automatically. Failure preserves artifacts and stops the queue.
"""
import argparse,datetime,fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent
CLI=['uvx','--with','six>=1.16,<2','--from','kaggle==2.2.5','kaggle']
PYTHON=sys.executable
OWNER='deeptanshukumar'
ARCHITECTURES=('gcn','gat','graphsage')
BANKS={
 'ogbn-arxiv':('afml-controlled-bank-20261108','kaggle-controlled-bank-20261108'),
 'ogbn-arxiv-temporal':('afml-temporal-bank-20261108','kaggle-temporal-bank-20261108'),
 'pair-align-mag-country':('afml-country-bank-20261108','kaggle-country-bank-20261108'),
}
STATE=ROOT/'RESEARCH_QUEUE.json'
QUEUE_CODE_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def atomic(path,data):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def mark(phase,**data):
    atomic(STATE,dict(phase=phase,updated_utc=now(),pid=os.getpid(),process_group=os.getpgrp(),queue_code_sha256=QUEUE_CODE_SHA256,**data));print(now(),phase,json.dumps(data),flush=True)
def run(args,*,read=False):
    for attempt in range(3 if read else 1):
        result=subprocess.run([str(x) for x in args],capture_output=True,text=True)
        if result.returncode==0:return result.stdout
        if read and attempt<2:time.sleep(15);continue
        raise RuntimeError('Command failed: '+repr([str(x) for x in args])+'\n'+result.stdout[-2500:]+'\n'+result.stderr[-2500:])
def wait_complete(slug):
    while True:
        output=run([*CLI,'kernels','status',OWNER+'/'+slug],read=True)
        if 'KernelWorkerStatus.COMPLETE' in output:return
        if any(x in output for x in ('KernelWorkerStatus.ERROR','KernelWorkerStatus.CANCEL','KernelWorkerStatus.FAILED')):raise RuntimeError(output)
        if not any(x in output for x in ('KernelWorkerStatus.RUNNING','KernelWorkerStatus.QUEUED')):raise RuntimeError('Unrecognized remote status: '+output)
        mark('waiting',job=slug,remote_status=output.strip());time.sleep(45)
def get_outputs(folder,slug):
    download=folder/'download-version-1';download.mkdir(exist_ok=True)
    proof=download/'LOCAL_STUDY_VERIFICATION.json'
    if not proof.exists():
        run([*CLI,'kernels','output',OWNER+'/'+slug+'/1','-p',download,'--quiet','--file-pattern',r'evidence/(BUNDLE.zip|STATUS.json|OUTPUT_HASHES.json|HARDWARE.json)$'],read=True)
        mark('verifying',job=slug)
        result=run([PYTHON,ROOT/'verify_study_outputs.py','--download',download,'--local-manifest',folder/'LOCAL_MANIFEST.json'])
        (folder/'LOCAL_VERIFICATION.log').write_text(result)
    data=json.loads(proof.read_text())
    if data.get('status')!='complete':raise ValueError('Local verification incomplete')
    return download

def verify_server(folder,slug):
    destination=folder/'server-version-1'
    if not destination.exists():
        run([*CLI,'kernels','pull',OWNER+'/'+slug+'/1','-p',destination,'-m'],read=True)
    metadata=json.loads((destination/'kernel-metadata.json').read_text())
    if metadata.get('is_private') is not True:raise ValueError('Remote job privacy not confirmed')
    expected=json.loads((folder/'LOCAL_MANIFEST.json').read_text())['script_sha256']
    scripts=list(destination.glob('*.py'))
    if len(scripts)!=1 or hashlib.sha256(scripts[0].read_bytes()).hexdigest()!=expected:raise ValueError('Remote script differs from prepared snapshot')
    atomic(folder/'SERVER_VERIFICATION.json',dict(status='complete',private=True,script_sha256=expected,verified_utc=now()))

def stage(mode,dataset,prior_slug,prior_download,*,architecture=None,gate=None):
    suffix={'ogbn-arxiv':'controlled','ogbn-arxiv-temporal':'temporal','pair-align-mag-country':'country'}[dataset]
    slug='afml-'+suffix+'-'+mode+('-'+architecture if architecture else '')+'-20261108'
    folder=ROOT/('kaggle-'+slug.removeprefix('afml-'))
    bank_slug,bank_folder=BANKS[dataset]
    if not folder.exists():
        quota=json.loads(run([*CLI,'quota','--format','json'],read=True))
        remaining=float(next(r for r in quota if r['resource']=='GPU')['remaining'].removesuffix('h'))
        required=9 if mode in {'extend','natural'} else 4 if mode=='final' and dataset=='ogbn-arxiv' else 2 if mode=='final' else 1
        if remaining<required:raise RuntimeError(f'Only {remaining} GPU hours remain; bounded {mode} job requests {required}. Queue stopped before allocation.')
        args=[PYTHON,ROOT/'prepare_research_stage.py','--mode',mode,'--dataset',dataset,'--bank-input',OWNER+'/'+bank_slug+'/1','--study-input',OWNER+'/'+prior_slug+'/1','--bank-proof',ROOT/bank_folder/'download-version-1/LOCAL_REPLAY_VERIFICATION.json','--study-proof',prior_download/'LOCAL_STUDY_VERIFICATION.json','--slug',slug,'--output-dir',folder]
        if architecture:args+=['--architecture',architecture]
        if gate:args+=['--phase-gate',gate]
        run(args)
        atomic(folder/'SUBMISSION_INTENT.json',dict(created_utc=now(),job=slug))
        manifest=json.loads((folder/'LOCAL_MANIFEST.json').read_text())
        # Mutations are issued exactly once. Ambiguous transport failures stop.
        receipt=run([*CLI,'kernels','push','-p',folder,'--timeout',manifest['timeout_seconds'],'--accelerator','NvidiaTeslaT4'])
        (folder/'SUBMISSION_RECEIPT.txt').write_text(receipt)
    elif not (folder/'SUBMISSION_RECEIPT.txt').exists():
        raise RuntimeError('Existing preparation without a submission receipt requires inspection: '+str(folder))
    verify_server(folder,slug);wait_complete(slug)
    return slug,get_outputs(folder,slug)

def merge(dataset,downloads):
    suffix={'ogbn-arxiv':'controlled','ogbn-arxiv-temporal':'temporal','pair-align-mag-country':'country'}[dataset]
    output=ROOT/('verified-'+suffix+'-final-20261108')
    if not (output/'PHASE_GATE.json').exists():run([PYTHON,ROOT/'merge_final_shards.py','--dataset',dataset,'--downloads',*downloads,'--output-dir',output])
    return output/'PHASE_GATE.json'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--through',choices=('design','extend','all'),default='all')
    args=parser.parse_args()
    with (ROOT/'RESEARCH_QUEUE.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            slug='afml-gcn-study-20261108';folder=ROOT/'kaggle-gcn-study-20261108'
            verify_server(folder,slug);wait_complete(slug);download=get_outputs(folder,slug)
            report=ROOT/'verified-design-20261108'
            if not report.exists():run([PYTHON,ROOT/'summarize_verified_design.py','--download',download,'--output-dir',report])
            if args.through=='design':mark('complete',through='design');return
            slug,download=stage('extend','ogbn-arxiv',slug,download)
            if args.through=='extend':mark('complete',through='extend');return
            controlled_slug,controlled_download=slug,download
            final_downloads=[stage('final','ogbn-arxiv',slug,download,architecture=arch)[1] for arch in ARCHITECTURES]
            gate4=merge('ogbn-arxiv',final_downloads)
            natural_gates=[]
            for dataset in ('ogbn-arxiv-temporal','pair-align-mag-country'):
                slug,download=stage('natural',dataset,controlled_slug,controlled_download,gate=gate4)
                outputs=[stage('final',dataset,slug,download,architecture=arch)[1] for arch in ARCHITECTURES]
                natural_gates.append(merge(dataset,outputs))
            combined=ROOT/'NATURAL_PHASE_GATE.json'
            audits=[json.loads(p.read_text()) for p in natural_gates]
            if not all(x.get('natural_final_complete') for x in audits):raise ValueError('Natural final audit incomplete')
            atomic(combined,dict(status='complete',experiment_5_complete=True,source_gates=[str(p) for p in natural_gates],controlled_gate=str(gate4)))
            stage('diagnostic','ogbn-arxiv',controlled_slug,controlled_download,gate=combined)
            mark('complete',through='all',note='All authorized execution stages have verified outputs; final scientific write-up still requires review.')
        except BaseException as error:
            mark('failed',error=str(error));raise

if __name__=='__main__':main()
