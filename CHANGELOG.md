# Changelog

## v3.0 (October 2026), standalone release

One shared `src/` and two runners, replacing the separate light and plus packages.

* **`spore_local.py`** (formerly `spore_light.py`) runs on a workstation and adds `--resume`.
* **`spore_cluster.py`** (new, replaces the SPORE+ express runner) runs one dataset per SLURM array task, reads `n_jobs` from `$SLURM_CPUS_PER_TASK`, turns figures off by default and writes its own sbatch script with `--print-sbatch`.
* **Stale-checkpoint fix.** Phase 9 now always overwrites its `_p9` files. Before, it skipped the write when a file already existed, so a run pointed at a used folder read the previous run's normalized data into Phase 10.
* **Protected list found from any folder.** A relative `protected_gene_list` that does not exist from the working directory is looked up next to the runner, so the same config works from the SPORE folder or from the FUNGI root.
* **Fresh-folder guard.** Both runners refuse to start if the output folders already hold `.h5ad` files, unless `--resume` is passed.
* **Configs from the runs that worked.** K562 and VCC h1-hESC configs follow the July HPC1 runs, including the 3,000,000-cell threshold that kept K562 in-process. The RPE1 config follows the local run. All use repository-relative paths and the `mbk_k` Phase 12 keys.
* **K562 protected list v2.0** (312 genes, including the 227 Tahoe drug-screen targets), replacing v1.0 (104 genes).
* `scikit-misc` added to `requirements.txt` (needed by Seurat v3 HVG selection).

## v2.1 (8 July 2026), one-stop rebuild

One command now produces the finished substrate and a report that proves it, with no manual patch scripts. The rebuild changed 9 files of the code that ran the July HPC1 jobs; the other 11 `src/` files are unchanged.

* **D1. Raw HVG panel on disk.** Phase 8 writes `{name}_{split}_hvgraw.h5ad`, raw counts on the selected genes, before Phase 9 normalizes.
* **D2. Automatic held-out coverage.** Phase 8 force-carries every validation and test target into the gene panel, in both the in-process and worker paths, and prints a coverage report. The worker's protected-list argument, which raised a `TypeError`, is fixed.
* **D3. Native Phase 12.** Phase 12 builds the control-preserving MiniBatchKMeans k=2 substrate and the train hybrid itself, with hard-fail guards, replacing a standalone builder script.
* **D4. Packaging.** A complete `requirements.txt`, and an absolute-import fallback in `phase08_hvg.py` so the worker subprocess no longer crashes on a relative import.
* **D5. Config validation.** SPORE checks the config against the data at startup. The batch key is read from the Phase 10 block. Gene IDs are harmonized offline from a symbol column when the file has one.
* **D6. CHITIN removed** from the code, configs and diagnostics.
* **D7. Verification gate.** The run ends by checking held-out coverage, the training firewall and the raw-count substrate, writes `COVERAGE_REPORT.{md,json}` and exits non-zero on failure.

## v2.0.1 (early July 2026), coverage recovery

Three fixes found when the downstream graph was missing three quarters of its held-out perturbations.

* Phase 2 exempts perturbation targets and protected genes from the ribosomal filter. Ribosomal knockdowns had been deleted before any graph was built.
* The protected list became split-independent, covering every surviving perturbation target.
* A stale-checkpoint bug was identified (later phases reused files from an earlier run in the same folder). It was worked around with fresh folders at the time and fixed in code in v3.0.

Held-out targets present in the final RPE1 graph rose from 35/152 to 119/152 on validation and from 79/304 to 249/304 on test.

## v2.0, SPORE+

Fourteen phases (0 to 13), driven by a notebook plus an express script for 32-CPU, 512 GB nodes. Added detection (Phase 1), separate ambient RNA and doublet phases, cell-line separation, the protected-gene tiers, worker subprocesses for the largest phases, and the experimental CHITIN phase.

## v1.0, first release

Notebook-driven, Phases 0 to 8. Sparse ingestion, cell triage, escaper filtering, gene triage, stratified zero-shot splits, HVG selection, normalization, confounder mitigation and optional metacells. Preserved on the `v1-archive` branch.
