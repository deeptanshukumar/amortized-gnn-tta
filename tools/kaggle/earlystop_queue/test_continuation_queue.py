"""Queue safety checks; fixtures do not submit jobs or represent research results."""
import importlib.util,json
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parent

def load(monkeypatch,tmp_path):
    spec=importlib.util.spec_from_file_location('continuation_queue',ROOT/'continue_research.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'STATE',tmp_path/'QUEUE.json')
    return module


def test_server_verification_rejects_public_or_changed_source(tmp_path,monkeypatch):
    m=load(monkeypatch,tmp_path)
    folder=tmp_path/'job';server=folder/'server-version-1';server.mkdir(parents=True)
    script=b'print("fixture")\n';sha=m.hashlib.sha256(script).hexdigest()
    (folder/'LOCAL_MANIFEST.json').write_text(json.dumps(dict(script_sha256=sha)))
    (server/'fixture.py').write_bytes(script)
    metadata=server/'kernel-metadata.json';metadata.write_text(json.dumps(dict(is_private=False)))
    with pytest.raises(ValueError,match='privacy'):m.verify_server(folder,'fixture')
    metadata.write_text(json.dumps(dict(is_private=True)))
    (server/'fixture.py').write_bytes(b'print("changed")\n')
    with pytest.raises(ValueError,match='differs'):m.verify_server(folder,'fixture')
    (server/'fixture.py').write_bytes(script);m.verify_server(folder,'fixture')
    assert json.loads((folder/'SERVER_VERIFICATION.json').read_text())['private'] is True


def test_incomplete_or_unknown_status_stops_without_submitting(tmp_path,monkeypatch):
    m=load(monkeypatch,tmp_path);calls=[]
    def fake(args,**kwargs):calls.append(args);return 'fixture KernelWorkerStatus.ERROR'
    monkeypatch.setattr(m,'run',fake)
    with pytest.raises(RuntimeError,match='ERROR'):m.wait_complete('fixture')
    assert len(calls)==1 and calls[0][-2:]==['status','deeptanshukumar/fixture']
    monkeypatch.setattr(m,'run',lambda *args,**kwargs:'unrecognized service response')
    with pytest.raises(RuntimeError,match='Unrecognized'):m.wait_complete('fixture')


def test_ambiguous_submission_is_never_repeated(tmp_path,monkeypatch):
    m=load(monkeypatch,tmp_path)
    folder=tmp_path/'kaggle-controlled-extend-trained100-20261109';folder.mkdir()
    monkeypatch.setattr(m,'run',lambda *args,**kwargs:pytest.fail('No remote call allowed'))
    with pytest.raises(RuntimeError,match='submission receipt'):m.stage('extend','ogbn-arxiv','fixture',tmp_path)


def test_read_retry_and_mutating_failure_have_different_behaviour(tmp_path,monkeypatch):
    m=load(monkeypatch,tmp_path);calls=[]
    class Failure:
        returncode=1;stdout='';stderr='fixture transport error'
    monkeypatch.setattr(m.subprocess,'run',lambda *args,**kwargs:(calls.append(args) or Failure()))
    monkeypatch.setattr(m.time,'sleep',lambda duration:None)
    with pytest.raises(RuntimeError):m.run(['fixture','push'])
    assert len(calls)==1
    calls.clear()
    with pytest.raises(RuntimeError):m.run(['fixture','status'],read=True)
    assert len(calls)==3


def test_insufficient_quota_stops_before_preparing_or_submitting(tmp_path,monkeypatch):
    m=load(monkeypatch,tmp_path)
    def quota_only(args,**kwargs):
        assert args[-3:]==['quota','--format','json']
        return json.dumps([dict(resource='GPU',remaining='3.00h')])
    monkeypatch.setattr(m,'run',quota_only)
    with pytest.raises(RuntimeError,match='requests 4'):m.stage('final','ogbn-arxiv','fixture',tmp_path,architecture='gcn')
    assert not (tmp_path/'kaggle-controlled-final-gcn-trained100-20261109').exists()
