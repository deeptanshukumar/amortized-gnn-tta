"""Source-ownership, exact-resume and interrupted-publication fixture checks."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch_geometric.data import Data
from amortized_gnn_tta.data.source_bank import formal_groups
from amortized_gnn_tta.data.api import graph_fingerprint
from amortized_gnn_tta.training import SourceTrainingData


def source_view():
    graph = Data(x=torch.randn(80, 8), edge_index=torch.stack((torch.arange(79), torch.arange(1, 80))), num_nodes=80)
    return SourceTrainingData(graph, torch.arange(48), torch.arange(48)%2,
                              torch.arange(48, 60), torch.arange(12)%2,
                              ('a','b'), dict(graph_sha256=graph_fingerprint(graph)))


def seed_options():
    return dict(split_seeds=list(range(71, 76)), initialization_seeds=[list(range(31+6*i, 37+6*i)) for i in range(5)])


def test_formal_groups_preserve_source_ownership_exact_class_budgets_and_native_ids():
    view = source_view()
    groups = formal_groups(view, torch.arange(80)+1000, **seed_options())
    assert [g['role'] for g, _ in groups] == ['meta_train']*3+['validation','test']
    assert len({s for g, _ in groups for s in g['initialization_seeds']}) == 30
    assert groups[0][0] == formal_groups(view, torch.arange(80)+1000, **seed_options())[0][0]
    assert len({tuple(g['train_indices']) for g, _ in groups}) == 5
    for group, data in groups:
        assert set(data.train_indices.tolist()) | set(data.val_indices.tolist()) == set(range(60))
        assert not torch.isin(data.train_indices, data.val_indices).any()
        assert torch.equal(data.train_labels.bincount(), view.train_labels.bincount())
        assert torch.equal(data.val_labels.bincount(), view.val_labels.bincount())
        assert group['train_native_ids'] == (data.train_indices+1000).tolist()
    bad = seed_options(); bad['initialization_seeds'][1][0] = 31
    with pytest.raises(ValueError, match='globally distinct'):
        formal_groups(view, torch.arange(80), **bad)


def test_formal_resume_recovers_orphan_detects_mutations_and_preserves_partial(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[1]/'examples'
    monkeypatch.syspath_prepend(str(examples))
    spec = importlib.util.spec_from_file_location('formal_runner', examples/'prepare_formal_bank.py')
    runner = importlib.util.module_from_spec(spec); spec.loader.exec_module(runner)
    view = source_view()
    monkeypatch.setattr(runner, 'load_bank_inputs', lambda a, p: ({'fixture':True}, view, torch.arange(80)))
    plan = dict(bank_kind='formal-real-v1',architectures=['gcn'],threads=1,device='cpu',
                source_epochs=1,source_patience=1,source_learning_rate=.01,source_weight_decay=.0005,
                hidden_channels=4,group_split_seeds=seed_options()['split_seeds'],
                group_initialization_seeds=seed_options()['initialization_seeds'])
    config = tmp_path/'config.json'; config.write_text(json.dumps(plan))
    a = SimpleNamespace(config=config,data_root=tmp_path,output_dir=tmp_path/'bank',fit=True,max_models=1)
    assert runner.execute(a, allow_fixture_cpu=True) == dict(completed=1,expected=30)
    index = a.output_dir/'bank.json'; saved=json.loads(index.read_text())
    # Simulate interruption after directory publication, before index update.
    index.write_text(json.dumps(dict(saved, models=[])))
    a.fit=False
    assert runner.execute(a, allow_fixture_cpu=True)['completed'] == 1
    assert json.loads(index.read_text())['models'] == saved['models']
    # Interrupt before completed checkpoint publication; previous fits survive.
    a.fit=True
    monkeypatch.setattr(runner, 'fit_source', lambda *args, **kw: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        runner.execute(a, allow_fixture_cpu=True)
    assert json.loads(index.read_text())['models'] == saved['models']
    assert (a.output_dir/'INTERRUPTION.json').exists()
    plan['source_learning_rate']=.02; config.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='identity mismatch'):
        runner.execute(a, allow_fixture_cpu=True)
    plan['source_learning_rate']=.01; config.write_text(json.dumps(plan))
    path = a.output_dir/saved['models'][0]['path']
    with path.open('ab') as handle: handle.write(b'tamper')
    with pytest.raises(ValueError, match='checkpoint identity mismatch'):
        runner.execute(a, allow_fixture_cpu=True)


def test_formal_diversity_fixture_reports_pair_metrics_without_target_labels(tmp_path, monkeypatch):
    examples = Path(__file__).resolve().parents[1]/'examples'
    monkeypatch.syspath_prepend(str(examples))
    import prepare_formal_bank as runner
    import audit_formal_bank as auditor
    view = source_view()
    inputs = lambda a, p: ({'fixture':True}, view, torch.arange(80))
    monkeypatch.setattr(runner, 'load_bank_inputs', inputs)
    monkeypatch.setattr(auditor, 'load_bank_inputs', inputs)
    plan = dict(bank_kind='formal-real-v1',architectures=['gcn'],threads=1,device='cpu',
                source_epochs=1,source_patience=1,source_learning_rate=.01,source_weight_decay=.0005,
                hidden_channels=4,group_split_seeds=seed_options()['split_seeds'],
                group_initialization_seeds=seed_options()['initialization_seeds'])
    config=tmp_path/'config.json'; config.write_text(json.dumps(plan))
    a=SimpleNamespace(config=config,data_root=tmp_path,output_dir=tmp_path/'bank',fit=True,max_models=2)
    runner.execute(a,allow_fixture_cpu=True)
    monkeypatch.setattr(auditor, 'execute', lambda a: runner.execute(a,allow_fixture_cpu=True))
    a.bank=a.output_dir; a.fixture=True; a.probe_seed=901; a.probe_nodes=32
    report=auditor.audit(a)
    assert report['final_test_access'] is False
    assert set(report['probe_indices']) <= set(view.train_indices.tolist())
    pair=report['architectures']['gcn']['pairs'][0]
    assert pair['relative_weight_l2'] > 0
    assert 0 <= pair['prediction_disagreement'] <= 1
    assert 0 <= pair['hidden_linear_cka'] <= 1
    assert report['architectures']['gcn']['models'] == 2


def test_formal_generators_require_matching_diversity_audit(tmp_path, monkeypatch):
    examples=Path(__file__).resolve().parents[1]/'examples'
    monkeypatch.syspath_prepend(str(examples))
    import run_arxiv_experiments as study
    bank=tmp_path/'bank'; bank.mkdir()
    (bank/'bank.json').write_text(json.dumps(dict(status='complete',plan={'bank_kind':'formal-real-v1'})))
    a=SimpleNamespace(bank=bank,output_dir=tmp_path/'study')
    with pytest.raises(ValueError, match='diversity audit before generator use'):
        study.Study(a)
    (bank/'DIVERSITY.json').write_text(json.dumps(dict(bank_sha256='stale')))
    with pytest.raises(ValueError, match='does not match the current bank'):
        study.Study(a)
