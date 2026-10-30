"""Portable packaging and device smoke checks; no cloud execution is claimed."""

import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def test_notebook_has_plain_python_and_no_saved_execution():
    notebook = json.loads((ROOT / "notebooks" / "colab_adaptation.ipynb").read_text())
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None and cell["outputs"] == []
            ast.parse("".join(cell["source"]))


def test_code_bundle_allowlist_hashes_and_no_overwrite(tmp_path):
    output = tmp_path / "bundle.zip"
    command = [sys.executable, str(ROOT / "examples" / "export_colab_bundle.py"), "--output", str(output)]
    subprocess.run(command, check=True, capture_output=True, text=True)
    with zipfile.ZipFile(output) as archive:
        metadata = json.loads(archive.read("amortized-gnn-tta/colab_bundle.json"))
        assert metadata["bank"] is None
        for name, expected in metadata["files_sha256"].items():
            assert not any(part in name.split("/") for part in (".git", ".venv", "references", "runs"))
            assert hashlib.sha256(archive.read("amortized-gnn-tta/" + name)).hexdigest() == expected
        assert "notebooks/colab_adaptation.ipynb" in metadata["files_sha256"]
    repeated = subprocess.run(command, capture_output=True, text=True)
    assert repeated.returncode != 0 and "choose a new path" in repeated.stderr
    incomplete = subprocess.run(command + ["--bank", "missing"], capture_output=True, text=True)
    assert incomplete.returncode != 0 and "all three" in incomplete.stderr


def test_actual_cpu_device_smoke_command(tmp_path):
    output = tmp_path / "device"
    result = subprocess.run([sys.executable, str(ROOT / "examples" / "check_adaptation_device.py"),
                             "--device", "cpu", "--output-dir", str(output)],
                            check=True, capture_output=True, text=True)
    summary = json.loads(result.stdout)
    report = json.loads((output / "device_report.json").read_text())
    assert summary["checks"] == len(report["checks"]) == 18
    assert report["device"] == "cpu" and report["gpu"] is None
    assert all(row["finite_outputs"] and row["cpu_reload_predictions_close"] for row in report["checks"])
