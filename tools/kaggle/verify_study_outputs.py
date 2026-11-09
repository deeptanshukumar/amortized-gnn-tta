"""Verify complete private study bundles and preserve their result inventory."""
import argparse,hashlib,itertools,json,shutil,zipfile
from pathlib import Path

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--download',type=Path,required=True);p.add_argument('--local-manifest',type=Path,required=True)
a=p.parse_args();statuses=list(a.download.rglob('STATUS.json'))
if len(statuses)!=1:raise ValueError('Expected one execution status')
evidence=statuses[0].parent;status=json.loads(statuses[0].read_text())
if status['state']!='complete':raise ValueError('Study did not complete: '+json.dumps(status))
if (evidence/'BUNDLE.zip').exists():
    with zipfile.ZipFile(evidence/'BUNDLE.zip') as z:
        for info in z.infolist():
            rel=Path(info.filename)
            if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe archive member')
            target=evidence/rel
            if target.exists():continue
            target.parent.mkdir(parents=True,exist_ok=True)
            with z.open(info) as source,target.open('xb') as destination:shutil.copyfileobj(source,destination,1024*1024)
hashes=json.loads((evidence/'OUTPUT_HASHES.json').read_text())
inventory={str(path.relative_to(evidence)) for path in evidence.rglob('*') if path.is_file() and path.name not in {'STATUS.json','OUTPUT_HASHES.json'}}
if set(hashes)!=inventory:raise ValueError('Output hash inventory is incomplete or contains missing files')
for name,value in hashes.items():
    rel=Path(name)
    if rel.is_absolute() or '..' in rel.parts:raise ValueError('Unsafe hash member')
    if digest(evidence/rel)!=value:raise ValueError('Output hash mismatch: '+name)
local=json.loads(a.local_manifest.read_text());source=evidence/'STUDY_SOURCE.zip'
if digest(source)!=local['study_source_sha256']:raise ValueError('Study source snapshot mismatch')
roots=[evidence/name for name in ('controlled-study','study') if (evidence/name).is_dir()]
if len(roots)!=1:raise ValueError('Expected one study root')
study=roots[0];fits=[]
for f in sorted(study.glob('*/RESULT.json')):
    row=json.loads(f.read_text())
    if row.get('final_test_scored') is not False:raise ValueError('Unexpected fitting label access')
    if not 0<=row['mean_validation_accuracy']<=1:raise ValueError('Invalid fitting accuracy')
    mean=sum(r['adapted_accuracy'] for r in row['records'])/len(row['records'])
    if abs(mean-row['mean_validation_accuracy'])>1e-12:raise ValueError('Summary/record inconsistency')
    fits.append(dict(name=row['name'],architecture=row['architecture'],conditioning=row['conditioning'],update=row['update'],method=row['method'],seed=row['seed'],accuracy=row['mean_validation_accuracy'],selected_epoch=row['selected_epoch'],fit_seconds=row['generator_fit_seconds'],records=len(row['records'])))
mode=local.get('mode','design')
inputs=json.loads((study/'INPUTS.json').read_text())
plan=inputs['plan']
bank=json.loads((evidence/'formal-bank/bank.json').read_text())
if bank['plan']!=plan:raise ValueError('Study and bank plans differ')
indexed={r['id']:r for r in bank['models']}
manifest=json.loads((evidence/'formal-bank/environment_manifest.json').read_text())
validation_envs={e['id'] for e in manifest['environments'] if e['role']=='validation'}
pins=json.loads((study/'SOURCE_PINS.json').read_text())
if any(indexed.get(mid)!=pin for mid,pin in pins.items()):raise ValueError('Study source pins differ from bank')
for row in fits:
    data=json.loads((study/row['name']/'RESULT.json').read_text())
    if len(data['records'])!=len({(r['model_id'],r['environment_id']) for r in data['records']}):raise ValueError('Duplicate validation cases')
    validation_models={r['id'] for r in bank['models'] if r['architecture']==row['architecture'] and r['role']=='validation'}
    if {(r['model_id'],r['environment_id']) for r in data['records']}!=set(itertools.product(validation_models,validation_envs)):raise ValueError('Validation case inventory incomplete')
    if row['selected_epoch']<0 or row['selected_epoch']>plan['generator_epochs']:raise ValueError('Invalid selected epoch')
if mode in {'design','extend','natural'}:
    replay=json.loads((study/'INDEPENDENT_REPLAY.json').read_text())
    if replay['status']!='complete':raise ValueError('Independent replay incomplete')
    expected=sum(r['records'] for r in fits)
    control=study/'CONDITIONING_CONTROL/COMPLETE.json'
    if control.exists():expected+=sum(len(r['records']) for r in json.loads(control.read_text())['rows'])
    if replay['cases']!=expected:raise ValueError('Replay inventory is incomplete')
else:replay={}
if mode=='design':
    if not (study/'EXPERIMENT_1.json').exists() or not (study/'EXPERIMENT_2.json').exists():raise ValueError('Design study incomplete')
    if not (study/'BASELINES-gcn.json').exists() or not (study/'CONDITIONING_CONTROL/COMPLETE.json').exists():raise ValueError('Required controls missing')
    e1=json.loads((study/'EXPERIMENT_1.json').read_text())
    if [r['conditioning'] for r in e1['rows']]!=list('ABCD'):raise ValueError('Incomplete input ablation')
    best=max(e1['rows'],key=lambda r:r['mean_validation_accuracy'])
    if e1['selected_inputs']!=best['conditioning']:raise ValueError('Input selection differs from declared tie rule')
    e2=json.loads((study/'EXPERIMENT_2.json').read_text())
    if {r['update'] for r in e2['rows']}!={'head','lora'}:raise ValueError('Incomplete update comparison')
if mode in {'design','extend','natural'}:
    for arch in ('gcn',) if mode=='design' else plan['architectures']:
        baseline=json.loads((study/f'BASELINES-{arch}.json').read_text())
        if baseline['final_test_scored'] is not False:raise ValueError('Baseline used final labels')
        models={r['id'] for r in bank['models'] if r['architecture']==arch and r['role']=='validation'}
        methods={'frozen','constant_teacher','practice_fixed','gtrans_feature_only','iterative_im_head','iterative_im_lora'}
        expected=set(itertools.product(models,validation_envs,methods))
        actual={(r['model_id'],r['environment_id'],r['method']) for r in baseline['rows']}
        if actual!=expected or len(baseline['rows'])!=len(expected):raise ValueError('Baseline case inventory incomplete')
if mode in {'extend','natural'} and not (study/'FROZEN_DESIGN.json').exists():raise ValueError('Design not frozen')
if mode in {'extend','natural'}:
    frozen=json.loads((study/'FROZEN_DESIGN.json').read_text())
    if frozen['identity']!=inputs or frozen['candidate_seeds']!=plan['candidate_seeds']:raise ValueError('Frozen design input mismatch')
    if set(frozen['selections'])!=set(plan['architectures']):raise ValueError('Architecture selection incomplete')
    experiment=json.loads((study/('EXPERIMENT_3.json' if mode=='extend' else 'EXPERIMENT_5_VALIDATION.json')).read_text())
    wanted=set(itertools.product(plan['architectures'],plan['generator_seeds'],('hypernetwork','teacher_hypernetwork','diffusion')))
    actual={(r['architecture'],r['seed'],r['method']) for r in experiment['rows']}
    if actual!=wanted or len(experiment['rows'])!=len(wanted):raise ValueError('Generator comparison inventory incomplete')
    for row in experiment['rows']:
        selected=frozen['selections'][row['architecture']]
        if any(row[k]!=selected[k] for k in ('conditioning','update')):raise ValueError('Comparison uses an unselected recipe')
        if json.loads((study/row['name']/'RESULT.json').read_text())!=row:raise ValueError('Experiment result differs from fit artifact')
if mode=='final':
    complete=json.loads((study/'FINAL/COMPLETE.json').read_text())
    if complete['architecture_shard']!=local['architecture']:raise ValueError('Wrong final architecture shard')
    evaluation=json.loads((study/'FINAL/EVALUATION_PLAN.json').read_text())
    if digest(evidence/'evaluate_final_shard.py')!=evaluation['audit_export_runner_sha256']:raise ValueError('Audit runner identity mismatch')
    scores=json.loads((study/'FINAL/SCORES.json').read_text())
    envs=json.loads((study/'FINAL/ENVIRONMENTS.json').read_text())
    targets={e['id'] for e in envs['environments'] if e['role']=='target_test'}
    models={r['id'] for r in bank['models'] if r['architecture']==local['architecture'] and r['role']=='test'}
    methods={'frozen','constant_teacher','practice_fixed','gtrans_feature_only','iterative_im_head','iterative_im_lora'}|{f'{m}-{s}' for m in ('hypernetwork','teacher_hypernetwork','diffusion') for s in plan['generator_seeds']}
    expected=set(itertools.product(models,targets,methods))
    actual={(r['model_id'],r['environment_id'],r['method']) for r in scores['rows']}
    if actual!=expected or len(scores['rows'])!=len(expected):raise ValueError('Final case inventory incomplete')
    if complete['design_sha256']!=digest(study/'FROZEN_DESIGN.json'):raise ValueError('Final frozen design mismatch')
    audits=json.loads((study/'FINAL/REPLAY_AUDIT.json').read_text())['rows']
    if len(audits)!=complete['independent_delta_replays']:raise ValueError('Final replay count mismatch')
    audited={(r['source_id'],r['environment_id'],r['method']) for r in audits}
    expected_audits=set(itertools.product(models,targets,methods-{'frozen','practice_fixed','gtrans_feature_only'}))
    if audited!=expected_audits or len(audits)!=len(audited):raise ValueError('Final weight replay inventory incomplete')
    if sum(r['prediction_class_differences'] for r in audits)!=complete['prediction_class_differences']:raise ValueError('Final replay difference count mismatch')
    for row in scores['rows']:
        if not 0<=row['accuracy']<=1 or abs(row['gain']-(row['accuracy']-row['baseline']))>1e-12:raise ValueError('Invalid final metric')
        if not (study/'FINAL'/f"{row['model_id']}--{row['environment_id']}--{row['method']}.pt").is_file():raise ValueError('Final prediction artifact missing')
if mode=='diagnostic':
    diagnostic=study/'DIAGNOSTIC'
    result=json.loads((diagnostic/'RESULTS.json').read_text())
    if result['final_test_scored'] is not False:raise ValueError('Diagnostic accessed final labels')
    manifest=json.loads((evidence/'formal-bank/environment_manifest.json').read_text())
    families=('control','attribute','structural','combined')
    cases={next(e['id'] for e in manifest['environments'] if e['role']=='meta_train' and e['family']==family) for family in families}
    methods={'supervised_head','supervised_lora','supervised_full','hypernetwork','teacher_hypernetwork','diffusion'}
    expected=set(itertools.product(plan['architectures'],cases,methods))
    actual={(r['architecture'],r['environment_id'],r['method']) for r in result['rows']}
    if actual!=expected or len(result['rows'])!=len(expected):raise ValueError('Diagnostic comparison incomplete')
    audit=json.loads((diagnostic/'REPLAY_AUDIT.json').read_text())
    if audit['cases']!=len(expected) or len(audit['rows'])!=len(expected):raise ValueError('Diagnostic replay incomplete')
    if len({r['artifact'] for r in audit['rows']})!=len(expected):raise ValueError('Duplicate diagnostic replay')
    audit_plan=json.loads((diagnostic/'PLAN.json').read_text())
    if audit_plan['audit_runner_sha256']!=digest(evidence/'diagnose_real_audit.py'):raise ValueError('Diagnostic audit runner mismatch')
    if audit_plan['design_sha256']!=digest(study/'FROZEN_DESIGN.json'):raise ValueError('Diagnostic frozen design mismatch')
    if any(not (diagnostic/r['artifact']).is_file() for r in audit['rows']):raise ValueError('Diagnostic prediction artifact missing')
proof=dict(status='complete',mode=mode,dataset=plan.get('dataset','ogbn-arxiv'),architecture_shard=local.get('architecture'),
           final_cases=len(scores['rows']) if mode=='final' else None,
           diagnostic_replay_cases=audit['cases'] if mode=='diagnostic' else None,
           study_source_sha256=digest(source),verified_files=len(hashes),fits=fits,
           frozen_design=(study/'FROZEN_DESIGN.json').exists(),
           frozen_design_sha256=digest(study/'FROZEN_DESIGN.json') if (study/'FROZEN_DESIGN.json').exists() else None,
           independent_replay_cases=replay.get('cases'),prediction_class_differences=replay.get('prediction_class_differences'),
           final_test_scored=status.get('final_test_scored',False),study_relative_path=str(study.relative_to(a.download)))
(a.download/'LOCAL_STUDY_VERIFICATION.json').write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps(dict(status='complete',mode=mode,verified_files=len(hashes),generator_fits=len(fits),replays=replay.get('cases'),frozen_design=proof['frozen_design'])))
