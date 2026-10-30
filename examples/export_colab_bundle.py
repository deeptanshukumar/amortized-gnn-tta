"""Export an explicit project allowlist and OPTIONAL audited synthetic bank.

Default exports code/config/tests/docs only. To use an existing source bank in
Colab, pass --bank, --model-partition AND --environments. No arbitrary runs,
reference repositories, virtual environments, credentials or user-home files
are scanned. ZIP artifacts are intended for runs/, never Git staging.
"""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import torch

from amortized_gnn_tta.bank_artifacts import artifact_path, load_model_partition, load_verified_bank
from amortized_gnn_tta.data import CSBMConfig, CSBMEnvironment, load_dataset, load_environment_manifest
from amortized_gnn_tta.models import build_model
from amortized_gnn_tta.training import source_training_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--model-partition", type=Path)
    parser.add_argument("--environments", type=Path)
    args = parser.parse_args()
    if any((args.bank, args.model_partition, args.environments)) and not all((args.bank, args.model_partition, args.environments)):
        parser.error("Provide all three bank/partition/environment inputs, or none")
    if args.output.exists():
        parser.error("Output ZIP exists; choose a new path")
    root = Path(__file__).resolve().parents[1]
    files = {}
    for folder, pattern in (("src", "*.py"), ("examples", "*.py"), ("tests", "*.py"), ("configs", "*.json"),
                            ("notebooks", "*.ipynb")):
        for path in sorted((root / folder).rglob(pattern)):
            if not path.resolve().is_relative_to(root):
                raise ValueError("Refusing a source symlink outside the repository")
            files[str(path.relative_to(root))] = path
    for name in ("pyproject.toml", "README.md", "THIRD_PARTY_NOTICES.md"):
        files[name] = root / name
    for name, path in files.items():
        if name.endswith(".ipynb"):
            notebook = json.loads(path.read_text(encoding="utf-8"))
            if any(cell.get("outputs") or cell.get("execution_count") is not None
                   for cell in notebook["cells"] if cell["cell_type"] == "code"):
                raise ValueError("Clear notebook outputs/execution counts before exporting")
    metadata = {"protocol": "colab-project-bundle-v1", "bank": None,
                "model_partition": None, "environments": None}
    if args.bank:
        torch.set_num_threads(1)
        raw = json.loads((args.bank / "source_subsets.json").read_text(encoding="utf-8"))
        values = dict(raw["source_provenance"]["graph_metadata"]["config"])
        values["split_ratios"] = tuple(values["split_ratios"])
        source_id = raw["source"]["graph_id"]
        source = source_training_data(load_dataset([
            CSBMEnvironment(source_id, "source_train", CSBMConfig(**values))]), source_id)
        bank = load_verified_bank(args.bank, source, build_model)
        load_model_partition(args.model_partition, bank)
        env = load_environment_manifest(args.environments)
        if env["reference"] != json.loads(json.dumps(values)):
            raise ValueError("Environment source reference disagrees with the bank")
        for name in ("bank.json", "source_subsets.json"):
            files[f"runs/uploaded-bank/{name}"] = bank.directory / name
        for row in bank.manifest["models"]:
            for field in ("checkpoint", "metrics"):
                files[f"runs/uploaded-bank/{row[field]}"] = artifact_path(bank.directory, row[field])
        files["runs/uploaded-model-partition.json"] = args.model_partition
        files["runs/uploaded-environments.json"] = args.environments
        metadata.update(bank="runs/uploaded-bank", model_partition="runs/uploaded-model-partition.json",
                        environments="runs/uploaded-environments.json", bank_sha256=bank.sha256)
    # Capture once: edits during compression cannot make checksums describe
    # different bytes from those actually included in the ZIP.
    contents = {name: path.read_bytes() for name, path in files.items()}
    metadata["files_sha256"] = {name: hashlib.sha256(content).hexdigest() for name, content in contents.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in contents.items():
            archive.writestr(f"amortized-gnn-tta/{name}", content)
        archive.writestr("amortized-gnn-tta/colab_bundle.json", json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "files": len(files),
                      "includes_audited_bank": args.bank is not None,
                      "bytes": args.output.stat().st_size}, indent=2))


if __name__ == "__main__":
    main()
