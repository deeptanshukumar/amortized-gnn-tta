"""Readable design-stage findings from independently verified numerical artifacts."""
import argparse,csv,datetime,json,shutil,statistics
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--download',type=Path,required=True)
p.add_argument('--output-dir',type=Path,required=True)
p.add_argument('--refresh',action='store_true',help='Preserve previous derived report files before replacing them')
a=p.parse_args()
proof=json.loads((a.download/'LOCAL_STUDY_VERIFICATION.json').read_text())
if proof.get('status')!='complete' or proof.get('mode')!='design':raise ValueError('Verified design-stage evidence required')
study=a.download/proof['study_relative_path'];rows=[];case_rows=[]
for fit in proof['fits']:
    result=json.loads((study/fit['name']/'RESULT.json').read_text())
    records=result['records'];base=statistics.mean(r['baseline_accuracy'] for r in records)
    zero=all(r['delta_norm']==0 for r in records)
    row=dict(name=fit['name'],conditioning=fit['conditioning'],update=fit['update'],method=fit['method'],seed=fit['seed'],validation_cases=len(records),selected_epoch=fit['selected_epoch'],frozen_accuracy=base,selected_accuracy=result['mean_validation_accuracy'],gain_pp=100*(result['mean_validation_accuracy']-base),all_recorded_delta_norms_zero=zero,fit_seconds=result['generator_fit_seconds'],adaptation_parameters=result['adaptation_parameter_count'],generator_parameters=result['generator_parameter_count'])
    rows.append(row)
    case_rows.extend(dict(fit=fit['name'],**r) for r in records)
base_result=json.loads((study/'BASELINES-gcn.json').read_text())
if base_result['final_test_scored'] is not False:raise ValueError('Unexpected final labels')
baselines=base_result['rows'];paired={(r['model_id'],r['environment_id']):r['accuracy'] for r in baselines if r['method']=='frozen'}
base_summary=[]
for method in sorted({r['method'] for r in baselines}):
    selected=[r for r in baselines if r['method']==method]
    gains=[r['accuracy']-paired[(r['model_id'],r['environment_id'])] for r in selected]
    base_summary.append(dict(method=method,cases=len(selected),mean_accuracy=statistics.mean(r['accuracy'] for r in selected),mean_gain_pp=100*statistics.mean(gains),harm_fraction=statistics.mean(g<0 for g in gains),offline_practice_labels=method in {'practice_fixed','constant_teacher'}))
controls=json.loads((study/'CONDITIONING_CONTROL/COMPLETE.json').read_text())
e1=json.loads((study/'EXPERIMENT_1.json').read_text());e2=json.loads((study/'EXPERIMENT_2.json').read_text())
a.output_dir.mkdir(parents=True,exist_ok=a.refresh)
if a.refresh:
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    for name in ('SUMMARY.json','GENERATOR_FITS.csv','GENERATOR_VALIDATION_CASES.csv','BASELINE_SUMMARY.csv','DESIGN_RESULTS.md'):
        path=a.output_dir/name
        if path.exists():shutil.copyfile(path,path.with_name(path.name+'.before-refresh-'+stamp))
for name,items in [('GENERATOR_FITS.csv',rows),('GENERATOR_VALIDATION_CASES.csv',case_rows),('BASELINE_SUMMARY.csv',base_summary)]:
    with (a.output_dir/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
summary=dict(status='verified',study_source_sha256=proof['study_source_sha256'],frozen_mean_accuracy=rows[0]['frozen_accuracy'],selected_inputs=e2['selected_inputs'],selected_update=e2['selected_update'],generator_fits=rows,baselines=base_summary,offline_label_audit='Constant teacher and practice-fixed baselines both use permitted offline practice labels. The original raw baseline flag marks only practice_fixed; this derived report corrects that incomplete label annotation without changing scores or fitting.',independent_replay_cases=proof['independent_replay_cases'],prediction_class_differences=proof['prediction_class_differences'],total_recorded_generator_fit_seconds=sum(r['fit_seconds'] for r in rows),final_test_scored=False,source=str(study.resolve()))
(a.output_dir/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
lines=['# Verified GCN design-stage results','',f"Source snapshot: `{proof['study_source_sha256']}`.",'','These are 2018 validation results used for design selection. Final 2019–2020 labels have not been scored.','', '| Inputs | Update | Selected epoch | Mean validation accuracy | Gain over frozen (pp) | Recorded zero deltas | Training time (s) |','|---|---|---:|---:|---:|---|---:|']
for r in rows:lines.append(f"| {r['conditioning']} | {r['update']} | {r['selected_epoch']} | {100*r['selected_accuracy']:.3f}% | {r['gain_pp']:+.3f} | {r['all_recorded_delta_norms_zero']} | {r['fit_seconds']:.2f} |")
lines+=['',f"Total recorded generator fitting time: {summary['total_recorded_generator_fit_seconds']:.2f} seconds. Job startup, data checks, diagnostics, baselines, replay and packaging are additional. No online speedup follows from these fit times.",'',f"Validation selection: inputs **{e2['selected_inputs']}**, update **{e2['selected_update']}**, using the predeclared tie rules. A tie does not establish that one descriptor or update site is scientifically superior.",'','## Baseline comparisons','', '| Method | Cases | Mean validation accuracy | Mean gain (pp) | Harmful cases | Offline practice labels |','|---|---:|---:|---:|---:|---|']
for r in base_summary:lines.append(f"| {r['method']} | {r['cases']} | {100*r['mean_accuracy']:.3f}% | {r['mean_gain_pp']:+.3f} | {100*r['harm_fraction']:.1f}% | {r['offline_practice_labels']} |")
lines+=['','Both practice-fixed fitting and constant-teacher construction use permitted offline practice labels; neither uses online target labels. The raw baseline label flag only marks practice-fixed, so this report corrects that incomplete annotation. GTrans is our declared feature-only variant, not a full published-method reproduction. Negative gains remain in the comparison.','', '## Conditioning sensitivity','', '| Inputs | Original accuracy | Cyclic-donor accuracy | Prediction disagreement |','|---|---:|---:|---:|']
for r in controls['rows']:lines.append(f"| {r['conditioning']} | {100*r['mean_normal_accuracy']:.3f}% | {100*r['mean_donor_accuracy']:.3f}% | {100*r['mean_prediction_disagreement']:.3f}% |")
lines+=['','Changing descriptors for a zero-update checkpoint cannot establish useful graph sensitivity. This diagnostic does not retrain generators or change selection.','',f"Independent saved-update replay verified {proof['independent_replay_cases']} cases. Recorded argmax differences: {proof['prediction_class_differences']}. The replay uses actual saved factors/deltas rather than regenerating conditioning.",'','## Interpretation','', 'The completed hypernetwork configurations did not demonstrate adaptation gains on this real-Arxiv validation protocol. Earlier modest synthetic-CSBM development gains therefore have not yet transferred to these tested configurations. This is not a conclusion about every generator family or architecture; the three-architecture, three-seed teacher-hypernetwork and diffusion comparison is separate. Shared graph cases and overlapping bank groups are not independent datasets.','']
(a.output_dir/'DESIGN_RESULTS.md').write_text('\n'.join(lines))
print(json.dumps({k:v for k,v in summary.items() if k not in {'generator_fits','baselines','source'}}))
