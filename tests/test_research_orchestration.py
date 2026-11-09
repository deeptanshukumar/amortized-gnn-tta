"""Fail-closed orchestration checks; no research fits executed here."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def runner():
    path = (
        Path(__file__).resolve().parents[1] / "examples/run_real_research_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("research_pipeline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pipeline_stops_at_failed_child_and_preserves_evidence(tmp_path, monkeypatch):
    module = runner()
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / "bank.json").write_text(
        json.dumps(
            dict(
                status="complete",
                plan={"checkpoint_seeds": {"meta_train": [31], "validation": [34]}},
                models=[],
            )
        )
    )
    calls = []

    def fail(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(module.subprocess, "run", fail)
    args = SimpleNamespace(
        bank=bank,
        output_dir=tmp_path / "run",
        ogb_root=tmp_path / "ogb",
        mag_root=tmp_path / "mag",
        wait_for_bank=True,
        bank_timeout_seconds=1,
    )
    with pytest.raises(RuntimeError, match="experiments-1-3 failed"):
        module.run(args)
    journal = json.loads((args.output_dir / "JOURNAL.json").read_text())
    assert (
        len(calls) == 1
        and journal["status"] == "failed"
        and journal["stages"][0]["returncode"] == 7
    )
    assert (args.output_dir / "experiments-1-3.log").exists()
    with pytest.raises(FileExistsError):
        module.run(args)


def test_waiter_rejects_failed_bank(tmp_path):
    module = runner()
    (tmp_path / "bank.json").write_text(
        json.dumps(dict(status="failed", error="deliberate fixture failure"))
    )
    with pytest.raises(RuntimeError, match="deliberate fixture failure"):
        module.wait_for_bank(tmp_path, 1)
