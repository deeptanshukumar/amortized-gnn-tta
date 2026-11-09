"""Validation-only sensitivity of fitted A–D generators to target descriptors.

A deterministic cyclic donor map uses the next validation environment in saved
manifest order. The original selected inputs/update sites are not revised.
"""
import argparse
import json
from pathlib import Path
import torch
from run_arxiv_experiments import Study
from amortized_gnn_tta.adaptation.checkpoints import load_generator_checkpoint, atomic_json
from amortized_gnn_tta.adaptation.conditioning_controls import donor_conditioning_predictions
from amortized_gnn_tta.adaptation_artifacts import state_sha256
from amortized_gnn_tta.evaluation import score_logits


def run(a):
    a.stop_after=2
    study=Study(a)
    output=study.root/'CONDITIONING_CONTROL'
    if (output/'COMPLETE.json').exists():
        return json.loads((output/'COMPLETE.json').read_text())
    output.mkdir(exist_ok=False)
    e1=json.loads((study.root/'EXPERIMENT_1.json').read_text())
    envs=study.validation
    if len(envs)<2:
        raise ValueError('Need distinct validation donor environments')
    donors={env['id']:envs[(i+1)%len(envs)]['id'] for i,env in enumerate(envs)}
    atomic_json(output/'PLAN.json',dict(donors=donors,source_models='validation only',
                labels='Only original target validation labels score after donor-conditioned prediction',
                retraining=False,changes_design_selection=False,final_test_scored=False))
    results=[]
    for fit in e1['rows']:
        architecture=fit['architecture']
        sources=study.sources(architecture,'validation')
        example=next(iter(sources.values()))
        generator,descriptor,_=load_generator_checkpoint(study.root/fit['name']/'generator.pt',model=example,device=study.device)
        records=[]
        for record in fit['records']:
            eid=record['environment_id']; mid=record['model_id']; model=sources[mid]
            saved=torch.load(study.root/fit['name']/record['artifact'],map_location='cpu',weights_only=True)
            before=state_sha256(model.state_dict())
            if saved['source_state_sha256']!=before:
                raise ValueError('Original source checkpoint changed')
            target_bundle=study.loader(study.manifest,eid)
            target=target_bundle.unlabeled_input(eid)
            donor=study.loader(study.manifest,donors[eid]).unlabeled_input(donors[eid])
            prediction=donor_conditioning_predictions(model,target,donor,descriptor,generator,scope=saved['scope'],device=study.device)
            indices=saved['indices']
            logits=prediction.logits[indices].detach().cpu()
            labels=target_bundle.evaluation_labels(eid,'eval')
            accuracy=score_logits(logits,indices=torch.arange(len(indices)),labels=labels).accuracy
            classes=logits.argmax(-1)
            disagreement=float((classes!=saved['predicted_classes']).float().mean())
            artifact=output/(fit['name']+'--'+record['artifact'])
            torch.save(dict(source_id=mid,source_state_sha256=before,environment_id=eid,donor_environment_id=donors[eid],
                            indices=indices,scope=saved['scope'],delta=prediction.delta.cpu(),factor_vector=prediction.factor_vector.cpu() if prediction.factor_vector is not None else None,
                            predicted_classes=classes,prediction_logits=logits),artifact)
            if before!=state_sha256(model.state_dict()): raise ValueError('Diagnostic changed source weights')
            records.append(dict(model_id=mid,environment_id=eid,donor_environment_id=donors[eid],
                                normal_accuracy=record['adapted_accuracy'],donor_accuracy=accuracy,
                                accuracy_change=accuracy-record['adapted_accuracy'],prediction_disagreement=disagreement,
                                delta_change_l2=float((prediction.delta.cpu()-saved['delta']).norm()),artifact=artifact.name))
        results.append(dict(conditioning=fit['conditioning'],fit=fit['name'],records=records,
                            mean_normal_accuracy=sum(r['normal_accuracy'] for r in records)/len(records),
                            mean_donor_accuracy=sum(r['donor_accuracy'] for r in records)/len(records),
                            mean_prediction_disagreement=sum(r['prediction_disagreement'] for r in records)/len(records)))
        atomic_json(output/'RESULTS.json',dict(rows=results,final_test_scored=False))
    result=dict(rows=results,donors=donors,changes_design_selection=False,final_test_scored=False,
                interpretation='Fixed-generator descriptor mismatch sensitivity; not a retrained no-input baseline or a deployment method. Zero sensitivity does not establish useful conditioning.')
    atomic_json(output/'COMPLETE.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bank',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    run(p.parse_args())
