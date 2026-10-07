#!/usr/bin/env python3
"""
spore_cluster.py
────────────────
SPORE runner for SLURM clusters (the former SPORE+ express runner, rebuilt on the current src/).

It runs exactly the same Phase 0-12 pipeline, verification gate and outputs as spore_local.py,
and adds what a shared cluster needs: one dataset per SLURM array task, n_jobs taken from the
CPUs SLURM allocated, figures off by default, and a generator for the sbatch script.

Input is either one SPORE config, or a batch file that lists several:

  # batch.yaml
  datasets:
    - configs/spore_config_k562.yaml
    - configs/spore_config_rpe1.yaml
  slurm:
    cpus_per_task: 32
    mem_gb: 256
    time: "24:00:00"
    partition: cpuq
    max_concurrent: 2
    activate: "source /path/to/venv/bin/activate"

Usage:
  python spore_cluster.py --config batch.yaml --list                 # list datasets in a batch
  python spore_cluster.py --config batch.yaml --print-sbatch > submit_spore.sh
  sbatch submit_spore.sh                                              # one array task per dataset
  python spore_cluster.py --config batch.yaml --dataset-index 0      # run one dataset by hand
  python spore_cluster.py --config configs/spore_config_k562.yaml   # single config

EXIT CODES
  0  Success
  1  Config / argument error
  2  One or more datasets failed
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spore_local as spore   # shared pipeline: config loading, validation, run_pipeline, gate


def _read_batch(config_path: str):
    """Return (list of SPORE config paths, slurm settings) for a batch file or a single config."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if "datasets" not in raw:
        return [str(Path(config_path).resolve())], raw.get("slurm", {})
    base = Path(config_path).resolve().parent
    paths = []
    for entry in raw["datasets"]:
        p = Path(entry)
        paths.append(str(p if p.is_absolute() else (base / p).resolve()))
    return paths, raw.get("slurm", {})


def _print_sbatch(config_path: str, dataset_paths: list, slurm: dict, extra_flags: str) -> None:
    n = len(dataset_paths)
    max_conc = slurm.get("max_concurrent", n)
    array_str = f"0-{n - 1}%{max_conc}" if n > 1 else "0"
    log_dir = Path(config_path).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)   # SLURM will not create the --output dir itself
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name=spore",
        f"#SBATCH --array={array_str}",
        f"#SBATCH --cpus-per-task={slurm.get('cpus_per_task', 32)}",
        f"#SBATCH --mem={slurm.get('mem_gb', 256)}G",
        f"#SBATCH --time={slurm.get('time', '24:00:00')}",
        f"#SBATCH --output={log_dir}/spore_%A_%a.out",
        f"#SBATCH --error={log_dir}/spore_%A_%a.err",
    ]
    if slurm.get("partition"):
        lines.append(f"#SBATCH --partition={slurm['partition']}")
    lines += [
        "",
        f"# SPORE cluster run, generated {datetime.now():%Y-%m-%d %H:%M}",
        f"# Config: {Path(config_path).resolve()}",
        "# Datasets (array index: config):",
    ]
    lines += [f"#   {i}: {p}" for i, p in enumerate(dataset_paths)]
    lines += [
        "",
        "# One BLAS thread per process; SPORE parallelizes with joblib across n_jobs instead.",
        "export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
        "export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1",
        "",
        slurm.get("activate", "# activate your Python environment here"),
        f"cd {HERE}",
        "",
        f"python spore_cluster.py --config {Path(config_path).resolve()} "
        f"--dataset-index $SLURM_ARRAY_TASK_ID {extra_flags}".rstrip(),
    ]
    print("\n".join(lines))


def _run_one(config_path: str, args) -> bool:
    try:
        cfg = spore.load_config(config_path)
    except Exception as exc:
        print(f"ERROR loading config {config_path}: {exc}", file=sys.stderr)
        return False

    # Cluster defaults: use every CPU SLURM gave us, and skip figures unless asked for.
    n_jobs = args.n_jobs or int(os.environ.get("SLURM_CPUS_PER_TASK", 0) or 0)
    if n_jobs > 0:
        cfg.setdefault("runtime", {})["n_jobs"] = n_jobs
    if not args.figures:
        cfg.setdefault("plotting", {})["save_figures"] = False

    stale = spore.check_fresh_outputs(cfg, args.resume)
    if stale:
        print(f"ERROR: {config_path}: output dirs already hold .h5ad files from an earlier run "
              f"(first: {stale[0]}). Use fresh dirs or pass --resume.", file=sys.stderr)
        return False

    logger = spore._make_logger(cfg)
    logger.info(f"spore_cluster: config {config_path} | n_jobs {cfg['runtime'].get('n_jobs')}")
    try:
        spore.validate_config_against_data(cfg, logger)
    except spore.ConfigDataMismatch as exc:
        logger.error(f"config/data validation failed: {exc}")
        return False

    if args.subset_first:
        with spore._timed("SUBSETTER", logger):
            spore.run_subsetter(cfg, logger)
    return spore.run_pipeline(cfg, logger)


def _parse_args():
    p = argparse.ArgumentParser(prog="spore_cluster.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, metavar="YAML",
                   help="A SPORE config, or a batch file with a 'datasets' list")
    p.add_argument("--dataset-index", type=int, default=None, metavar="N",
                   help="Run dataset N of a batch (defaults to $SLURM_ARRAY_TASK_ID)")
    p.add_argument("--all", action="store_true", help="Run every dataset in the batch sequentially")
    p.add_argument("--list", action="store_true", help="List the datasets in the batch and exit")
    p.add_argument("--print-sbatch", action="store_true", help="Print a SLURM array script and exit")
    p.add_argument("--n-jobs", type=int, default=None,
                   help="Override n_jobs (default: $SLURM_CPUS_PER_TASK, else the config value)")
    p.add_argument("--figures", action="store_true", help="Save diagnostic figures (off by default)")
    p.add_argument("--subset-first", action="store_true",
                   help="Run the perturbation subsetter before the pipeline")
    p.add_argument("--resume", action="store_true",
                   help="Allow existing outputs in the configured dirs (continue a crashed run)")
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    if not Path(args.config).exists():
        print(f"ERROR: config not found: {args.config}", file=sys.stderr)
        return 1
    dataset_paths, slurm = _read_batch(args.config)

    if args.list:
        for i, p in enumerate(dataset_paths):
            print(f"{i}: {p}")
        return 0

    if args.print_sbatch:
        flags = " ".join(f for f, on in (("--figures", args.figures), ("--subset-first", args.subset_first),
                                         ("--resume", args.resume)) if on)
        _print_sbatch(args.config, dataset_paths, slurm, flags)
        return 0

    if args.all:
        selected = list(range(len(dataset_paths)))
    else:
        index = args.dataset_index
        if index is None:
            index = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
        if not 0 <= index < len(dataset_paths):
            print(f"ERROR: dataset index {index} out of range (0-{len(dataset_paths) - 1})", file=sys.stderr)
            return 1
        selected = [index]

    failed = [dataset_paths[i] for i in selected if not _run_one(dataset_paths[i], args)]
    if failed:
        print(f"FAILED: {failed}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
