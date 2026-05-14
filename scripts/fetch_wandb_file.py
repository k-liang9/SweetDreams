#!/usr/bin/env python3
"""Download a model checkpoint file from a Weights & Biases run."""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import wandb


DEFAULT_CANDIDATES = ("best_model.pt", "model.pt", "checkpoint.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch a checkpoint saved on a W&B run into a local path."
    )
    parser.add_argument(
        "run_path",
        help="W&B run path in the form entity/project/run_id.",
    )
    parser.add_argument(
        "--output",
        default="weights/vqvae.pt",
        help="Local output path to write. Defaults to weights/vqvae.pt.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Exact W&B run file to download. Defaults to best_model.pt with fallbacks.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Overwrite the output path if it already exists.",
    )
    return parser.parse_args()


def choose_file(run: wandb.apis.public.Run, requested: str | None) -> wandb.apis.public.File:
    files = list(run.files())
    by_name = {file.name: file for file in files}

    candidates = (requested,) if requested else DEFAULT_CANDIDATES
    for name in candidates:
        if name in by_name:
            return by_name[name]

    pt_files = [file for file in files if file.name.endswith(".pt")]
    best_files = [file for file in pt_files if Path(file.name).name == "best_model.pt"]
    if len(best_files) == 1:
        return best_files[0]
    if requested:
        available = ", ".join(sorted(file.name for file in files))
        raise FileNotFoundError(
            f"Run does not contain requested file {requested!r}. Available files: {available}"
        )
    if len(pt_files) == 1:
        return pt_files[0]

    available_pt = ", ".join(sorted(file.name for file in pt_files)) or "none"
    raise FileNotFoundError(
        "Could not infer which checkpoint to download. "
        f"Pass --file explicitly. Available .pt files: {available_pt}"
    )


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    if output_path.exists() and not args.replace:
        raise FileExistsError(f"{output_path} already exists. Pass --replace to overwrite.")

    api = wandb.Api()
    run = api.run(args.run_path)
    file = choose_file(run, args.file)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wandb-download-") as tmp_dir:
        downloaded = Path(file.download(root=tmp_dir, replace=True).name)
        shutil.copy2(downloaded, output_path)

    print(f"Downloaded {args.run_path}/{file.name} -> {output_path}")


if __name__ == "__main__":
    main()
