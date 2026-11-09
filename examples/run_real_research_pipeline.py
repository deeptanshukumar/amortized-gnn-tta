"""Serial, fail-closed orchestration of the approved real-data study.

This runner does not grant compute resources or promise completion within a
particular time. It logs each subprocess and stops at its first failure. Existing
artifacts are preserved. The source-bank waiter permits attaching to a bank
already running in another terminal, without starting duplicate fits.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from amortized_gnn_tta.adaptation.checkpoints import atomic_json


def wait_for_bank(directory, timeout_seconds):
    start = time.monotonic()
    while True:
        path = directory / "bank.json"
        if path.exists():
            bank = json.loads(path.read_text())
            if bank["status"] == "complete":
                return
            if bank["status"] == "failed":
                raise RuntimeError(
                    "Source bank failed: " + bank.get("error", "unknown error")
                )
        if time.monotonic() - start > timeout_seconds:
            raise TimeoutError("Existing source bank did not finish")
        time.sleep(10)


def run(args):
    for field in ("bank", "output_dir", "ogb_root", "mag_root"):
        setattr(args, field, getattr(args, field).resolve())
    root = Path(__file__).resolve().parents[1]
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    journal = []
    environment = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )

    def execute(name, script, arguments):
        command = [
            sys.executable,
            str(root / "examples" / script),
            *map(str, arguments),
        ]
        row = dict(
            name=name, command=command, status="running", started_unix=time.time()
        )
        journal.append(row)
        atomic_json(output / "JOURNAL.json", dict(stages=journal, status="running"))
        print("STAGE START", name, flush=True)
        with (output / (name + ".log")).open("x") as log:
            result = subprocess.run(
                command, cwd=root, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
        row.update(
            status="complete" if result.returncode == 0 else "failed",
            returncode=result.returncode,
            elapsed_seconds=time.time() - row["started_unix"],
        )
        atomic_json(
            output / "JOURNAL.json",
            dict(
                stages=journal, status="running" if result.returncode == 0 else "failed"
            ),
        )
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect its preserved log")
        print("STAGE DONE", name, row["elapsed_seconds"], flush=True)

    controlled = output / "controlled"
    common = [
        "--bank",
        args.bank,
        "--data-root",
        args.ogb_root,
        "--output-dir",
        controlled,
    ]
    if args.wait_for_bank:
        current = json.loads((args.bank / "bank.json").read_text())
        expected = {
            (role, seed)
            for role in ("meta_train", "validation")
            for seed in current["plan"]["checkpoint_seeds"][role]
        }
        present = {
            (row["role"], row["seed"])
            for row in current["models"]
            if row["architecture"] == "gcn"
        }
        if current["status"] == "running" and expected <= present:
            execute(
                "experiments-1-2",
                "run_arxiv_experiments.py",
                common + ["--stop-after", "2"],
            )
        print("Waiting for existing controlled source bank", flush=True)
        wait_for_bank(args.bank, args.bank_timeout_seconds)
    else:
        execute(
            "controlled-bank",
            "prepare_arxiv_bank.py",
            [
                "--config",
                root / "configs/arxiv_research_v1.json",
                "--data-root",
                args.ogb_root,
                "--output-dir",
                args.bank,
            ],
        )
    execute("experiments-1-3", "run_arxiv_experiments.py", common)
    execute(
        "experiment-4",
        "evaluate_real_generalization.py",
        common + ["--final-evaluation", "--measure-memory"],
    )
    execute(
        "controlled-report",
        "summarize_real_study.py",
        ["--bank", args.bank, "--study", controlled],
    )
    frozen = controlled / "FROZEN_DESIGN.json"
    for label, data_root, config in [
        ("temporal", args.ogb_root, "arxiv_temporal_research_v1.json"),
        ("country", args.mag_root, "mag_country_research_v1.json"),
    ]:
        bank = output / (label + "-bank")
        study = output / label
        execute(
            label + "-bank",
            "prepare_arxiv_bank.py",
            [
                "--config",
                root / "configs" / config,
                "--data-root",
                data_root,
                "--output-dir",
                bank,
            ],
        )
        natural = ["--bank", bank, "--data-root", data_root, "--output-dir", study]
        execute(
            label + "-generators",
            "run_arxiv_experiments.py",
            natural + ["--natural-design", frozen],
        )
        execute(
            label + "-final",
            "evaluate_real_generalization.py",
            natural + ["--final-evaluation", "--measure-memory"],
        )
        execute(
            label + "-report",
            "summarize_real_study.py",
            ["--bank", bank, "--study", study],
        )
    execute("experiment-6-diagnostic", "diagnose_real_adaptation.py", common)
    execute(
        "controlled-report-complete",
        "summarize_real_study.py",
        ["--bank", args.bank, "--study", controlled],
    )
    atomic_json(
        output / "JOURNAL.json",
        dict(
            stages=journal,
            status="complete",
            limitations=[
                "CPU only for the executed profile; no unexecuted GPU claim",
                "Fixed-budget model fits are not known optima",
                "Dataset download/construction and search add costs beyond final selected generator fits",
            ],
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--ogb-root", type=Path, required=True)
    p.add_argument("--mag-root", type=Path, required=True)
    p.add_argument("--wait-for-bank", action="store_true")
    p.add_argument("--bank-timeout-seconds", type=float, default=21600)
    args = p.parse_args()
    try:
        run(args)
    except BaseException as error:
        journal_path = args.output_dir / "JOURNAL.json"
        if journal_path.exists():
            evidence = json.loads(journal_path.read_text())
            evidence.update(status="failed", error=str(error))
            atomic_json(journal_path, evidence)
        raise
