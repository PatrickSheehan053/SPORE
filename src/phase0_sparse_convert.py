"""
SPORE_heavy · src/phase0_sparse_convert.py
──────────────────────────────────────────
Phase 0: Data Ingestion & Sparsification
  - Intelligent cache detection
  - Backed-mode dense reading
  - Chunked O(1) CSR sparsification
"""

import os
import anndata as ad
import scipy.sparse as sp
from .utils import log_phase_header, log_memory, force_gc

def run_phase0(cfg: dict, logger):
    log_phase_header(logger, 0, "Data Ingestion & Sparsification")
    raw_path = cfg['paths']['raw_h5ad']
    p0 = cfg.get("phase0_ingestion", {"chunk_size": 50000})

    # 1. Determine target sparse path
    if raw_path.endswith('_sparse.h5ad'):
        sparse_path = raw_path
    else:
        sparse_path = raw_path.replace('.h5ad', '_sparse.h5ad')

    # 2. Cache Hit: Sparse file already exists
    if os.path.exists(sparse_path):
        logger.info(f"  ✓ Sparse cache found: {sparse_path}")
        logger.info("  Loading directly into memory...")
        adata = ad.read_h5ad(sparse_path)
        
        # Double check it is actually sparse
        if not sp.issparse(adata.X):
            logger.warning("  ⚠ Matrix is NOT sparse! Enforcing CSR conversion...")
            adata.X = sp.csr_matrix(adata.X)
            
        cfg['paths']['raw_h5ad'] = sparse_path
        return adata

    # 3. Cache Miss: Perform chunked conversion
    logger.info(f"  No sparse cache found. Opening dense matrix in backed mode...")
    logger.info(f"  Source: {raw_path}")
    adata_backed = ad.read_h5ad(raw_path, backed='r')
    
    if sp.issparse(adata_backed.X):
        logger.info("  Matrix is already sparse on disk. Loading to memory...")
        adata = adata_backed.to_memory()
        adata_backed.file.close()
        cfg['paths']['raw_h5ad'] = raw_path
        return adata

    chunk_size = p0["chunk_size"]
    logger.info(f"  Matrix is dense. Converting to CSR in chunks of {chunk_size:,}...")
    
    sparse_chunks = []
    for i in range(0, adata_backed.n_obs, chunk_size):
        end = min(i + chunk_size, adata_backed.n_obs)
        dense_chunk = adata_backed.X[i:end] 
        sparse_chunks.append(sp.csr_matrix(dense_chunk))
        logger.info(f"    → Sparsified cells {i:,} to {end:,}")
        del dense_chunk
        force_gc(logger)

    logger.info("  Stacking chunks into final sparse matrix...")
    X_sparse = sp.vstack(sparse_chunks)
    
    logger.info("  Building in-memory AnnData object...")
    adata = ad.AnnData(
        X=X_sparse,
        obs=adata_backed.obs.copy(),
        var=adata_backed.var.copy(),
        uns=adata_backed.uns.copy() if adata_backed.uns else {},
        obsm=adata_backed.obsm.copy() if hasattr(adata_backed, 'obsm') and adata_backed.obsm else {}
    )
    
    adata_backed.file.close()
    del adata_backed
    del sparse_chunks
    force_gc(logger)

    logger.info(f"  Saving sparse dataset to disk: {sparse_path}")
    adata.write_h5ad(sparse_path)
    
    # Update config dynamically so downstream phases know where the data is
    cfg['paths']['raw_h5ad'] = sparse_path
    
    return adata
