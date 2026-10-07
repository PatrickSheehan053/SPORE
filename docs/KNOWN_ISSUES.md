# Known issues

Last reviewed 7 October 2026. Read this before the first run.

## 1. This rebuild has not finished an end-to-end run yet

An audit of HPC1 in October 2026 found exactly which code produced the July 2026 cluster runs (three VCC h1-hESC runs and two K562 runs, all successful). That code is the pre-rebuild snapshot of 8 July. This release is that snapshot plus the one-stop rebuild (fixes D1 to D7 in `CHANGELOG.md`), which changed 9 files: `diagnostics.py`, `phase01_detection.py`, `phase07_splits.py`, `phase08_hvg.py`, `phase10_confounders.py`, `phase12_metacell.py`, `reporting.py`, `utils.py` and the runner. The other 11 `src/` files are byte-identical to the code that ran.

The rebuild itself has never run on a full dataset, so no `COVERAGE_REPORT.md` exists yet. The first run is planned on VCC h1-hESC, compared against the July output, and its report will be committed under `docs/`.

If you run it first, check three things in the output. `COVERAGE_REPORT.md` should say `GRAPH-READY`, the train hybrid should contain no `val` or `test` units, and Phase 12 should log one control unit per control cell.

## 2. The large-dataset worker path is untested at scale

Above `runtime.large_dataset_threshold` cells, Phases 8 and 10 run in worker subprocesses. The first July K562 run crashed in that path (`Phase 8 subprocess failed`). The successful runs raised the threshold to 3,000,000 so both phases stayed in-process on a 256 GB node, and `configs/spore_config_k562.yaml` ships with that value. The rebuild fixes the worker's import and wires the protected list and held-out force-carry into it, but no real run has gone through the worker path since.

## 3. Batch correction now uses the configured key

The pre-rebuild code read the batch column from `dataset.batch_col` (default `gem_group`) and ignored `phase10_confounders.batch_correction.batch_key`. A config that named a different column, such as `batch` for VCC, silently ran on `gem_group` or fell back to uncorrected PCA. The rebuild uses `batch_key` and stops at startup if the column is missing. Phase 10 embeddings can therefore differ from the July runs for datasets whose batch column is not `gem_group`. The splits, the HVG panel and the Phase 12 substrate (built from raw counts) do not depend on Phase 10.

## 4. Ghost excision is skipped on the in-process path

Phase 10 temporarily adds cell-cycle marker genes to score the cell cycle and then removes them ("ghost excision"). It reads the core gene list from `{name}_train_p8.h5ad`. When that file is absent, which is the usual case on the in-process path, Phase 10 copies the Phase 9 files forward without removing the markers. The normalized `_p10` splits can then carry a few extra cell-cycle genes. The Phase 12 substrate and train hybrid are built from the Phase 8 raw HVG files and are not affected.

## 5. Example configs need checking against your data

* `configs/spore_config_k562.yaml` and `configs/spore_config_h1hesc.yaml` follow the July HPC runs. `configs/spore_config_rpe1.yaml` follows the local RPE1 run.
* Split sizes (`test_n`, `val_n`) are counts, not fractions. Scale them to the number of perturbations that survive Phase 5, which SPORE logs. SPORE stops if they leave no training perturbations.
* The perturbation column, control label and batch key follow the original files. SPORE checks all three at startup.

## 6. Thesis-era names inside the code

Log lines, comments and docstrings still mention the internal names from the thesis (HYPHAE, SPORE_light, SPORE+, `exp_0NN`, "Defect 1 to 7"). The user-facing run banners have been renamed. A full pass is a follow-up. `src/phase04_doublets.py` also prints a cluster-specific `activate` hint when Scrublet is missing.

## 7. Dead CHITIN plot helpers

CHITIN (Phase 13 in SPORE+) was removed, but a few of its plot helpers remain in `src/plotting.py`. Nothing calls them.

## 8. Not shipped

The SPORE+ notebook (`spore+.ipynb`) and its express runner call the removed CHITIN API and are superseded by `spore_cluster.py`. The first release (8 phases, notebook-driven) is preserved on the `v1-archive` branch.
