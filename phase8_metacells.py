"""
SPORE_heavy · src/phase8_metacells.py
──────────────────────────────────────
Phase 8: Meta-Cell Aggregation & Systema Calibration
  - Parallelized intra-perturbation K-Means pooling (Scatter-Gather)
  - Sharded Checkpointing (Auto-saves progress to disk)
  - Strict leakage firewall & Systema Calibration
"""

import os
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp
from sklearn.cluster import MiniBatchKMeans
from joblib import Parallel, delayed
from .utils import log_phase_header, snapshot, log_memory, force_gc

def _process_perturbation(pert, X_sub, emb_sub, target_k):
    """Isolated worker function. Processes a tiny slice of data in parallel."""
    # ── THE FIX: BREAK THE READ-ONLY LOCK ──
    # Joblib serializes data to workers using read-only memory maps to save RAM.
    # Scikit-learn's sparse Cython engine strictly requires writable memory buffers.
    # By copying these microscopic slices, we create writable arrays in local worker RAM.
    X_sub = X_sub.copy()
    emb_sub = emb_sub.copy()
    
    n_cells = X_sub.shape[0]
    n_metacells = max(1, n_cells // target_k)
    
    if n_metacells > 1:
        kmeans = MiniBatchKMeans(n_clusters=n_metacells, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(emb_sub)
    else:
        labels = np.zeros(n_cells, dtype=int)
        
    mc_exprs = []
    mc_obs = []
    
    for c in range(n_metacells):
        c_mask = (labels == c)
        if c_mask.sum() == 0:  # Safety guard for abandoned clusters
            continue
            
        if sp.issparse(X_sub):
            expr = np.asarray(X_sub[c_mask].mean(axis=0)).flatten()
        else:
            expr = X_sub[c_mask].mean(axis=0)
            
        mc_exprs.append(expr)
        mc_obs.append({"n_cells_in_metacell": c_mask.sum()})
        
    return pert, mc_exprs, mc_obs

def aggregate_split(adata, cfg: dict, logger, label: str):
    """Aggregates single cells into metacells using parallel sharding."""
    p8 = cfg["phase8_metacells"]
    n_jobs = cfg["runtime"]["n_jobs"]
    target_k = p8["target_cells_per_metacell"]
    pert_col = cfg["dataset"]["perturbation_col"]
    
    # ── Checkpoint Setup ──
    checkpoint_dir = os.path.join(cfg["paths"]["processed_dir"], "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # ── THE FIX: Pull dataset name and inject it into the shard prefix ──
    dataset_name = cfg["dataset"]["name"]
    shard_prefix = os.path.join(checkpoint_dir, f"{dataset_name}_shard_{label.lower()}")
    
    logger.info(f"  [{label}] Aggregating into meta-cells (target: {target_k} cells/mc)")
    
    # ── DYNAMIC EMBEDDING FALLBACK ──
    if hasattr(adata, 'obsm') and 'X_pca_harmony' in adata.obsm:
        use_rep = 'X_pca_harmony'
        logger.info(f"  [{label}] Clustering via Harmony PCA embeddings.")
    elif hasattr(adata, 'obsm') and 'X_pca' in adata.obsm:
        use_rep = 'X_pca'
        logger.info(f"  [{label}] Clustering via standard PCA embeddings.")
    else:
        logger.info(f"  [{label}] No PCA found. Calculating temporary PCA for rapid clustering...")
        import scanpy as sc
        # Calculate a rapid, memory-safe PCA just for the clustering step
        sc.pp.pca(adata, n_comps=50, use_highly_variable=False, zero_center=False, svd_solver='randomized')
        use_rep = 'X_pca'
        logger.info(f"  [{label}] Temporary PCA complete. Proceeding with fast clustering.")
        
    logger.info(f"  [{label}] Booting {n_jobs} parallel workers with shard checkpointing...")
    
    unique_perts = adata.obs[pert_col].unique()
    new_X_list = []
    new_obs_list = []
    
    # ── SHARDED EXECUTION ──
    chunk_size = 500  
    
    for i in range(0, len(unique_perts), chunk_size):
        pert_chunk = unique_perts[i:i + chunk_size]
        shard_file = f"{shard_prefix}_{i}.npz"
        
        if os.path.exists(shard_file):
            logger.info(f"  [{label}] Loading recovered shard: {i} to {i+len(pert_chunk)}")
            data = np.load(shard_file, allow_pickle=True)
            new_X_list.extend(data['X'])
            for pert_val, n_cells in zip(data['perts'], data['counts']):
                new_obs_list.append({pert_col: pert_val, "n_cells_in_metacell": n_cells})
            continue
            
        logger.info(f"  [{label}] Processing perturbations {i} to {i+len(pert_chunk)}...")
        
        tasks = []
        for pert in pert_chunk:
            pert_mask = (adata.obs[pert_col] == pert).values
            X_sub = adata.X[pert_mask]
            
            # Extract the correct coordinate space for K-Means based on our dynamic check
            emb_sub = adata.obsm[use_rep][pert_mask] if use_rep != 'X' else X_sub
            
            tasks.append((pert, X_sub, emb_sub, target_k))
            
        results = Parallel(n_jobs=n_jobs)(
            delayed(_process_perturbation)(*task) for task in tasks
        )
        
        shard_X, shard_perts, shard_counts = [], [], []
        for pert, exprs, obs_data in results:
            if not exprs: continue
            shard_X.extend(exprs)
            for o in obs_data:
                shard_perts.append(pert)
                shard_counts.append(o["n_cells_in_metacell"])
                
        np.savez(shard_file, X=np.array(shard_X), perts=np.array(shard_perts), counts=np.array(shard_counts))
        
        new_X_list.extend(shard_X)
        for pert_val, n_cells in zip(shard_perts, shard_counts):
            new_obs_list.append({pert_col: pert_val, "n_cells_in_metacell": n_cells})
            
        force_gc(logger)
        
    X_meta = np.vstack(new_X_list)
    obs_meta = pd.DataFrame(new_obs_list)
    
    adata_meta = ad.AnnData(X=X_meta, obs=obs_meta, var=adata.var.copy())
    if hasattr(adata, 'uns'):
        adata_meta.uns = adata.uns.copy()
    
    logger.info(f"  [{label}] Compressed {adata.n_obs:,} cells → {adata_meta.n_obs:,} meta-cells")
    return adata_meta

def calculate_systema_centroids(adata_meta, cfg: dict, logger):
    pert_col = cfg["dataset"]["perturbation_col"]
    ctrl_label = cfg["dataset"]["control_label"]
    
    logger.info("  Calculating Systema geometric centroids (Training Split Only)...")
    
    ctrl_mask = (adata_meta.obs[pert_col] == ctrl_label).values
    if ctrl_mask.sum() == 0:
        return None, None
        
    C_ctrl = adata_meta.X[ctrl_mask].mean(axis=0)
    
    unique_perts = set(adata_meta.obs[pert_col].unique()) - {ctrl_label}
    pert_centroids = []
    
    for pert in unique_perts:
        p_mask = (adata_meta.obs[pert_col] == pert).values
        if p_mask.sum() > 0:
            pert_centroids.append(adata_meta.X[p_mask].mean(axis=0))
            
    if not pert_centroids:
        return C_ctrl, None
        
    O_pert = np.vstack(pert_centroids).mean(axis=0)
    logger.info("  Systema calibration successful. C_ctrl and O_pert vectors established.")
    return C_ctrl, O_pert

def run_phase8(splits: dict, cfg: dict, logger):
    p8 = cfg.get("phase8_metacells", {})
    if not p8.get("enabled", False):
        logger.info("  Phase 8 Meta-Cell Aggregation: DISABLED")
        return splits

    log_phase_header(logger, 8, "Meta-Cell Aggregation & Systema Calibration")
    meta_splits = {}
    
    for key in ["train", "val", "test"]:
        meta_splits[key] = aggregate_split(splits[key], cfg, logger, key.title())
        force_gc(logger)
        
    if p8.get("systema_calibration", True):
        C_ctrl, O_pert = calculate_systema_centroids(meta_splits["train"], cfg, logger)
        if C_ctrl is not None and O_pert is not None:
            for key in ["train", "val", "test"]:
                meta_splits[key].uns["systema_C_ctrl"] = C_ctrl
                meta_splits[key].uns["systema_O_pert"] = O_pert
                
    for key in ["train", "val", "test"]:
        snapshot(meta_splits[key], f"Post Phase 8 ({key})", logger)

    return meta_splits