"""
SPORE · src/phase1_cell_triage.py
──────────────────────────────────
Phase 1: Cell-Level Triage
  - Mitochondrial (MT) filtering
  - Ribosomal gene removal
  - Library size / sparsity bounds

Memory-safe: uses boolean masks and a single safe_subset call.
"""

import numpy as np
import scanpy as sc
from collections import OrderedDict
from .utils import log_phase_header, snapshot, ensure_sparse, safe_subset, \
    log_memory, force_gc


def compute_qc_metrics(adata, cfg: dict, logger):
    """
    Annotate QC metrics on adata.obs:
      - pct_counts_mt, n_genes_by_counts, total_counts
    Also ensures the matrix is sparse (critical for 60+ GB datasets).
    """
    log_phase_header(logger, 1, "Cell-Level Triage")

    # ── CRITICAL: Ensure sparse IMMEDIATELY ────────────────────────────────
    adata = ensure_sparse(adata, logger)
    log_memory(logger, "after sparse conversion")

    p1 = cfg["phase1_cell_triage"]

    # Identify mitochondrial genes
    if p1["mt_method"] == "explicit_list":
        mt_genes = p1["mt_genes"]
        adata.var["mt"] = adata.var_names.isin(mt_genes)
        n_found = adata.var["mt"].sum()
        logger.info(f"  MT method: explicit_list → {n_found}/13 MT genes found")
    else:
        prefix = "MT-" if cfg["dataset"]["organism"] == "human" else "mt-"
        adata.var["mt"] = adata.var_names.str.startswith(prefix)
        logger.info(f"  MT method: prefix '{prefix}' → {adata.var['mt'].sum()} genes")

    # Compute QC metrics (works natively on sparse matrices)
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None,
                                log1p=False, inplace=True)
    logger.info(f"  QC metrics computed (pct_counts_mt, n_genes_by_counts, total_counts)")
    snapshot(adata, "Pre-filter", logger)
    return adata


def filter_cells(adata, cfg: dict, logger):
    """
    Apply Phase 1 filters using boolean masks, then a SINGLE safe_subset call.
    No intermediate .copy() calls — this is what prevents OOM.
    """
    from collections import OrderedDict
    import numpy as np

    # ── PERMANENT QC METRIC FIREWALL ─────────────────────────────────────────
    # If the loaded matrix is completely raw, these metrics won't exist.
    # We calculate them natively here so the pipeline NEVER crashes.
    if "total_counts" not in adata.obs.columns or "pct_counts_mt" not in adata.obs.columns:
        logger.info("  Raw matrix detected: Calculating QC metrics (total_counts, pct_counts_mt, n_genes_by_counts)...")
        import scanpy as sc
        adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    # ─────────────────────────────────────────────────────────────────────────

    p1 = cfg["phase1_cell_triage"]
    waterfall = OrderedDict()
    waterfall["Starting cells"] = adata.n_obs

    # Build cell mask incrementally
    keep_cells = np.ones(adata.n_obs, dtype=bool)

    # 1. Mitochondrial filter
    mt_thresh = p1["mt_threshold"]
    before = keep_cells.sum()
    keep_cells &= (adata.obs["pct_counts_mt"].values <= mt_thresh * 100)
    logger.info(f"  MT filter (≤{mt_thresh*100:.0f}%): removed {before - keep_cells.sum():,} cells")
    waterfall["After MT filter"] = int(keep_cells.sum())

    # 2. Min genes per cell
    before = keep_cells.sum()
    keep_cells &= (adata.obs["n_genes_by_counts"].values >= p1["min_genes_per_cell"])
    logger.info(f"  Min genes (≥{p1['min_genes_per_cell']}): removed {before - keep_cells.sum():,}")
    waterfall["After min genes"] = int(keep_cells.sum())

    # 3. Max genes per cell
    before = keep_cells.sum()
    keep_cells &= (adata.obs["n_genes_by_counts"].values <= p1["max_genes_per_cell"])
    logger.info(f"  Max genes (≤{p1['max_genes_per_cell']:,}): removed {before - keep_cells.sum():,}")
    waterfall["After max genes"] = int(keep_cells.sum())

    # 4. Min counts per cell
    before = keep_cells.sum()
    keep_cells &= (adata.obs["total_counts"].values >= p1["min_counts_per_cell"])
    logger.info(f"  Min counts (≥{p1['min_counts_per_cell']}): removed {before - keep_cells.sum():,}")
    waterfall["After min counts"] = int(keep_cells.sum())

    # 5. Max counts per cell
    before = keep_cells.sum()
    keep_cells &= (adata.obs["total_counts"].values <= p1["max_counts_per_cell"])
    logger.info(f"  Max counts (≤{p1['max_counts_per_cell']:,}): removed {before - keep_cells.sum():,}")
    waterfall["After max counts"] = int(keep_cells.sum())

    # Build gene mask (ribosomal removal)
    ribo_prefixes = tuple(p1["ribo_prefixes"])
    keep_genes = ~adata.var_names.str.startswith(ribo_prefixes)
    n_ribo = (~keep_genes).sum()
    logger.info(f"  Ribosomal gene removal: {n_ribo:,} genes stripped ({', '.join(p1['ribo_prefixes'])})")

    log_memory(logger, "before subset")

    # ── SINGLE subset operation — no intermediate copies ───────────────────
    adata_new = safe_subset(adata, cell_mask=keep_cells, gene_mask=keep_genes,
                            logger=logger)

    # Free the old adata
    del adata
    force_gc(logger)

    snapshot(adata_new, "Post Phase 1", logger)
    return adata_new, waterfall


def run_phase1(adata, cfg: dict, logger):
    """Full Phase 1 pipeline: compute QC → filter → return (adata, waterfall)."""
    adata = compute_qc_metrics(adata, cfg, logger)
    adata, waterfall = filter_cells(adata, cfg, logger)
    return adata, waterfall