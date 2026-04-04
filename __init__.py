"""
SPORE · src/
─────────────
Systematic Preprocessing and Optimization for Robust Evaluation
"""

from .utils import load_config, setup_logger, log_phase_header, snapshot
from .utils import ensure_sparse, safe_subset, log_memory, force_gc
from .plotting import apply_spore_style

from . import phase1_cell_triage
from . import phase2_escaper_filtering
from . import phase3_gene_triage
from . import phase4_splits
from . import phase5_hvg
from . import phase6_normalization
from . import phase7_confounders
from . import plotting
