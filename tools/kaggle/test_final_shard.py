"""External audit-runner checks; no benchmark accuracies or GPU claims."""
import importlib.util
from pathlib import Path
import pytest
import torch

ROOT=Path(__file__).resolve().parent

def load():
    spec=importlib.util.spec_from_file_location('final_shard_audit',ROOT/'evaluate_final_shard.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_near_tie_is_recorded_without_changing_original_scores(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT.parent/'amortized-gnn-tta-generator-integration/examples'))
    module=load()
    original=torch.tensor([[1.0,1.0+1e-7]])
    replay=torch.tensor([[1.0+1e-7,1.0]])
    result=module.check_replayed_logits(original,replay)
    assert result['prediction_class_differences']==1
    assert original.argmax(-1).item()==1
    with pytest.raises(AssertionError):module.check_replayed_logits(original,replay+0.1)


@pytest.mark.parametrize('architecture',['gcn','gat','graphsage'])
def test_saved_delta_roundtrip_with_complete_tiny_final_fixture(tmp_path,monkeypatch,architecture):
    project=ROOT.parent/'amortized-gnn-tta-generator-integration'
    monkeypatch.syspath_prepend(str(project/'examples'))
    original_loader=importlib.util.spec_from_file_location
    def choose(name,path,*args,**kwargs):
        if Path(path).name=='evaluate_real_generalization.py':path=ROOT/'evaluate_final_shard.py'
        if Path(path).name=='diagnose_real_adaptation.py':path=ROOT/'diagnose_real_audit.py'
        return original_loader(name,path,*args,**kwargs)
    monkeypatch.setattr(importlib.util,'spec_from_file_location',choose)
    spec=original_loader('original_runner_fixture',project/'tests/test_arxiv_runner.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    namespace=module.SimpleNamespace
    def shard_namespace(**kwargs):
        if 'final_evaluation' in kwargs:kwargs['architecture']=architecture
        return namespace(**kwargs)
    monkeypatch.setattr(module,'SimpleNamespace',shard_namespace)
    module.test_real_study_fits_restores_and_replays_both_generator_families(tmp_path,monkeypatch,architecture)
    if architecture!='gcn':
        # The upstream fixture deliberately returns after generator replay for
        # non-GCN models. Complete the separate export checks here, without
        # invoking its GCN-only A-D selection routine.
        spec=original_loader('shard_core_fixture',project/'examples/run_arxiv_experiments.py')
        core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
        args=namespace(bank=tmp_path/'bank',output_dir=tmp_path/'study',data_root=tmp_path,
                       final_evaluation=True,warmups=0,repeats=1,measure_memory=False,architecture=architecture)
        study=core.Study(args)
        study.baselines(architecture,dict(conditioning='A',update='lora'))
        core.atomic_json(study.root/'FROZEN_DESIGN.json',dict(identity=study.identity,
                         fitting_source_pins=dict(study.pins),selections={architecture:dict(conditioning='A',update='lora')},
                         candidate_seeds=study.plan['candidate_seeds'],final_test_scored=False))
        monkeypatch.setitem(__import__('sys').modules,'run_arxiv_experiments',core)
        for name in ('evaluate_final_shard.py','diagnose_real_audit.py'):
            spec=original_loader('external_'+name.removesuffix('.py'),ROOT/name)
            runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner);runner.run(args)
    assert (tmp_path/'study/FINAL/REPLAY_AUDIT.json').exists()
    complete=__import__('json').loads((tmp_path/'study/FINAL/COMPLETE.json').read_text())
    assert complete['architecture_shard']==architecture
    diagnostic=__import__('json').loads((tmp_path/'study/DIAGNOSTIC/REPLAY_AUDIT.json').read_text())
    assert diagnostic['cases']==24
    assert diagnostic['prediction_class_differences']==0
