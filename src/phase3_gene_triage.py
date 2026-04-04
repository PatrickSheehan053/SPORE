"""
SPORE · src/phase3_gene_triage.py
──────────────────────────────────
Phase 3: Gene-Level Triage
  - Ambient RNA / strict noise filtering
  - Scale-adaptive expression filter
  - Perturbation target rescue override

Memory-safe: Uses .getnnz() to bypass boolean matrix allocation, and
a pure-numpy C-buffer column reconstructor to bypass SciPy subsetting bloat.
"""

import numpy as np
import anndata as ad
import scipy.sparse as sp
from collections import OrderedDict
from .utils import log_phase_header, snapshot, log_memory, force_gc

def safe_in_memory_gene_subset(adata, keep_mask, logger):
    """
    Destructively filters columns (genes) of a CSR matrix IN-PLACE. 
    Uses a row-by-row C-buffer shift to strictly enforce O(1) memory allocation,
    completely avoiding the massive RAM bloat of vectorized subsetting.
    """
    if keep_mask.all():
        return adata
        
    logger.info("  Applying ultra-low-RAM in-place gene (column) mutation...")
    
    n_kept_genes = keep_mask.sum()
    
    if sp.issparse(adata.X) and adata.X.format == "csr":
        indptr = adata.X.indptr
        indices = adata.X.indices
        data = adata.X.data
        
        # 1. Create a mapping from old column index to new column index.
        # Dropped columns are marked as -1.
        col_mapping = np.zeros(adata.X.shape[1], dtype=np.int32) - 1
        col_mapping[keep_mask] = np.arange(n_kept_genes, dtype=np.int32)
        
        new_indptr = np.zeros_like(indptr)
        write_ptr = 0
        
        # 2. Row-by-row shift guarantees zero massive array allocations
        # Because we only drop data, write_ptr is always <= start, 
        # meaning we never overwrite unread data.
        for i in range(len(indptr) - 1):
            start = indptr[i]
            end = indptr[i+1]
            new_indptr[i] = write_ptr
            
            if start < end:
                r_ind = indices[start:end]
                r_dat = data[start:end]
                
                # Map old genes to new genes; keep only valid ones (>= 0)
                m_cols = col_mapping[r_ind]
                mask = m_cols >= 0
                nnz_row = mask.sum()
                
                if nnz_row > 0:
                    # Shift valid elements leftward in the master buffer
                    indices[write_ptr : write_ptr + nnz_row] = m_cols[mask]
                    data[write_ptr : write_ptr + nnz_row] = r_dat[mask]
                    
                write_ptr += nnz_row
                
        new_indptr[-1] = write_ptr
        
        # 3. Create fresh CSR from the truncated buffers
        new_X = sp.csr_matrix(
            (data[:write_ptr], indices[:write_ptr], new_indptr), 
            shape=(adata.n_obs, n_kept_genes)
        )
    else:
        logger.warning("  Matrix is not CSR. Falling back to default slicing.")
        new_X = adata.X[:, keep_mask]

    # 4. Subset metadata cleanly
    new_var = adata.var.iloc[keep_mask].copy()
    new_obs = adata.obs.copy()
    
    new_obsm = {k: v.copy() for k, v in adata.obsm.items()} if hasattr(adata, 'obsm') else {}
    new_varm = {k: v[keep_mask] for k, v in adata.varm.items()} if hasattr(adata, 'varm') else {}
    new_uns = adata.uns.copy() if hasattr(adata, 'uns') else {}
        
    del adata
    force_gc(logger)
    
    # 5. Rebuild AnnData wrapper to satisfy internal validators
    return ad.AnnData(
        X=new_X, obs=new_obs, var=new_var,
        obsm=new_obsm, varm=new_varm, uns=new_uns
    )


def _get_perturbation_targets(adata, cfg: dict) -> set:
    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    targets = set(adata.obs[pert_col].unique()) - {ctrl_label}
    return targets


def filter_genes(adata, cfg: dict, logger):
    log_phase_header(logger, 3, "Gene-Level Triage")
    p3 = cfg["phase3_gene_triage"]
    waterfall = OrderedDict()
    waterfall["Starting genes"] = adata.n_vars

    X = adata.X
    log_memory(logger, "Before feature metric calculation")
    
    if sp.issparse(X):
        # FIX: getnnz() bypasses the massive boolean matrix allocation of (X > 0)
        cells_per_gene = X.getnnz(axis=0)
        mean_expr = np.array(X.sum(axis=0)).flatten() / adata.n_obs
    else:
        cells_per_gene = (X > 0).sum(axis=0)
        mean_expr = X.mean(axis=0)

    keep_genes = np.ones(adata.n_vars, dtype=bool)

    # ── 1. Ambient RNA filter ──────────────────────────────────────────────
    min_cells = p3["min_cells_expressing"]
    ambient_fail = cells_per_gene < min_cells
    n_ambient = ambient_fail.sum()
    keep_genes &= ~ambient_fail
    logger.info(f"  Ambient RNA filter (≥{min_cells} cells): {n_ambient:,} genes flagged")
    waterfall["After ambient filter"] = int(keep_genes.sum())

    # ── 2. Scale-adaptive filter ───────────────────────────────────────────
    n_cells = adata.n_obs
    if n_cells < p3["small_dataset_threshold"]:
        pct_thresh = p3["pct_filter"]
        min_cells_adaptive = int(n_cells * pct_thresh)
        adaptive_fail = cells_per_gene < min_cells_adaptive
        logger.info(f"  Scale-adaptive: small dataset mode "
                    f"(≥{pct_thresh*100:.0f}% = {min_cells_adaptive:,} cells)")
    else:
        umi_thresh = p3["mean_umi_threshold"]
        adaptive_fail = mean_expr < umi_thresh
        logger.info(f"  Scale-adaptive: large dataset mode "
                    f"(mean UMI ≥ {umi_thresh})")

    new_removals = adaptive_fail & keep_genes
    n_adaptive = new_removals.sum()
    logger.info(f"  Adaptive filter: {n_adaptive:,} additional genes flagged")

    would_remove = ~keep_genes | adaptive_fail
    waterfall["After adaptive (pre-rescue)"] = int((~would_remove).sum())

    # ── 3. Perturbation target rescue ──────────────────────────────────────
    rescued = []
    if p3["rescue_perturbation_targets"]:
        targets = _get_perturbation_targets(adata, cfg)
        var_names = set(adata.var_names)
        targets_in_features = targets & var_names

        gene_name_to_idx = {g: i for i, g in enumerate(adata.var_names)}
        targets_to_rescue = {t for t in targets_in_features
                             if would_remove[gene_name_to_idx[t]]}

        if targets_to_rescue:
            rescued = sorted(targets_to_rescue)
            for t in targets_to_rescue:
                would_remove[gene_name_to_idx[t]] = False
            logger.info(f"  ⚡ Target rescue: {len(rescued)} perturbation targets saved")
        else:
            logger.info(f"  Target rescue: no perturbation targets at risk")

    final_keep = ~would_remove
    waterfall["After adaptive (post-rescue)"] = int(final_keep.sum())

    # Safely reconstruct the columns
    adata_new = safe_in_memory_gene_subset(adata, keep_mask=final_keep, logger=logger)
    
    snapshot(adata_new, "Post Phase 3", logger)
    return adata_new, waterfall, rescued

def run_phase3(adata, cfg: dict, logger):
    return filter_genes(adata, cfg, logger)
