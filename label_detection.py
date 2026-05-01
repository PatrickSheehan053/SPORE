"""
SPORE / CHITIN Config Detective
────────────────────────────────
Automatically inspects a raw .h5ad file and reports the most likely
values for perturbation_col, control_label, and batch_col.

USAGE:
    1. Change H5AD_PATH to your file.
    2. Run:  python label_detection.py
       OR:   python label_detection.py /path/to/file.h5ad

No other modifications needed.
"""

# ─────────────────────────────────────────────────────────────────────
# ▼▼▼  ONLY CHANGE THIS LINE  ▼▼▼
H5AD_PATH = "/scratch/patrick.sheehan/MYCELIUM/SPORE_heavy/raw_h5ads/multi_cancer_perturb_seq.h5ad"
# ▲▲▲  ONLY CHANGE THIS LINE  ▲▲▲
# ─────────────────────────────────────────────────────────────────────

import sys
import re
import numpy as np
from pathlib import Path
from collections import Counter

try:
    import scanpy as sc
except ImportError:
    sys.exit("ERROR: scanpy not found.")

# ── Vocabulary tables ─────────────────────────────────────────────────────────

# Ordered by descending priority. perturbation_name is first because it is
# the most unambiguous name for a per-gene perturbation column.
PERT_COL_VOCAB = [
    "perturbation_name", "perturbation", "pert_name", "pert",
    "gene", "gene_name", "target", "target_gene", "target_id",
    "crispr_target", "ko_gene", "kd_gene", "knockout", "knockdown",
    "pert_gene", "gene_target", "perturbed_gene", "intervention",
    # lower priority — these often contain guide IDs not gene names
    "treatment", "sgrna", "sgrna_name", "guide", "guide_id", "guide_name",
    "feature_name", "barcode_identity", "grna_gene", "sgid_ab",
    "gRNA_maxCount_identity", "condition",
]

# Ordered by priority — more specific terms first, generic "control" last.
# ntc is placed low because it appears inside guide IDs like sgNTC_1_1
CTRL_VOCAB_ORDERED = [
    "non-targeting",
    "nontargeting",
    "non_targeting",
    "non-targeting control",
    "negative_control",
    "neg_ctrl",
    "negative control",
    "unperturbed",
    "untreated",
    "mock",
    "scramble",
    "scrambled",
    "safe-targeting",
    "safetargeting",
    "wt",
    "wildtype",
    "wild_type",
    "gfp",
    "luciferase",
    "lacz",
    "empty",
    "control",   # generic — only matched if nothing above was found
    "ctrl",
    "ntc",       # lowest priority — too often a substring of guide IDs
]

BATCH_COL_VOCAB = [
    "gem_group", "gem_id", "batch", "batch_id", "replicate", "rep",
    "plate", "plate_id", "lane", "library", "library_id",
    "donor", "sample", "sample_id", "run", "sequencing_run",
    "10x_channel", "channel", "chip_id",
]

ORGANISM_HINTS = {
    "human": ["homo sapiens", "human", "hg38", "grch38", "grch37", "hg19", "hsapiens"],
    "mouse": ["mus musculus", "mouse", "mm10", "grcm38", "grcm39", "mmusculus"],
}

SEP = "─" * 70


# ── Helpers ───────────────────────────────────────────────────────────────────

def _col_lower(name):
    return name.lower().replace("-", "_").replace(" ", "_")


def _score_col(name, vocab):
    nl = _col_lower(name)
    for i, v in enumerate(vocab):
        if nl == v:
            return 1000 - i   # exact match: higher score for earlier vocab entries
    for i, v in enumerate(vocab):
        if v in nl or nl in v:
            return 500 - i
    for i, v in enumerate(vocab):
        for token in nl.split("_"):
            if token == v:
                return 200 - i
    return 0


def _looks_like_pert_column(series, min_unique=5):
    """
    Returns (bool, reason).
    Rejects boolean flag columns (2-3 unique values like 'perturbed'/'control')
    and columns whose values don't resemble gene symbols.
    """
    uniq = series.dropna().astype(str).unique()
    n_uniq = len(uniq)

    if n_uniq < min_unique:
        return False, f"only {n_uniq} unique values — likely a boolean flag column"

    sample = uniq[:50]
    gene_like = sum(
        1 for v in sample
        if len(v) <= 25 and re.match(r'^[A-Za-z0-9][A-Za-z0-9\-_\.]*$', v)
    )
    if gene_like / max(len(sample), 1) < 0.4:
        return False, "values don't resemble gene symbols"

    return True, "ok"


def _find_control_label(series):
    """
    Find the control label using ordered vocabulary.
    Uses exact full-value match first (prevents sgNTC_1_1 matching 'ntc'),
    then substring match only for high-priority terms, never for 'ntc'/'ctrl'.
    """
    uniq = series.dropna().astype(str).unique().tolist()

    # Pass 1: exact full-value match against ordered vocabulary
    for v in CTRL_VOCAB_ORDERED:
        hits = [u for u in uniq if _col_lower(u) == _col_lower(v)]
        if hits:
            return hits[0], "exact_match"

    # Pass 2: substring match — but only for high-priority unambiguous terms
    # Explicitly skip 'ntc', 'ctrl', 'control' in substring pass to avoid
    # matching guide IDs like sgNTC_1_1 or sgControl_2
    HIGH_PRIORITY = [
        "non-targeting", "nontargeting", "non_targeting",
        "negative_control", "neg_ctrl", "unperturbed", "untreated",
        "mock", "scramble", "safe-targeting",
    ]
    for v in HIGH_PRIORITY:
        hits = [u for u in uniq if v in _col_lower(u)]
        if hits:
            # Prefer shorter matches (less likely to be guide IDs)
            hits.sort(key=len)
            return hits[0], "substring_match"

    return None, "not_found"


def _top_pert_candidates(obs_cols, series_map, n=6):
    scored = [(c, _score_col(c, PERT_COL_VOCAB)) for c in obs_cols]
    scored = [(c, s) for c, s in scored if s > 0]
    scored.sort(key=lambda x: -x[1])
    return scored[:n]


def _top_batch_candidates(obs_cols, n=4):
    scored = [(c, _score_col(c, BATCH_COL_VOCAB)) for c in obs_cols]
    scored = [(c, s) for c, s in scored if s > 0]
    scored.sort(key=lambda x: -x[1])
    return scored[:n]


def _detect_organism(adata):
    haystack = ""
    for k in ["organism", "species", "genome", "reference_genome", "reference"]:
        haystack += str(adata.uns.get(k, "")).lower() + " "
    for c in adata.var.columns:
        haystack += c.lower() + " "
    haystack += " ".join(adata.var_names[:20]).lower()
    for org, hints in ORGANISM_HINTS.items():
        for h in hints:
            if h in haystack:
                return org, h
    mt_human = sum(1 for g in adata.var_names[:500] if g.startswith("MT-"))
    mt_mouse = sum(1 for g in adata.var_names[:500] if g.startswith("mt-"))
    if mt_human > mt_mouse:
        return "human", "MT- prefix heuristic"
    if mt_mouse > mt_human:
        return "mouse", "mt- prefix heuristic"
    return "human", "default_fallback"


def _detect_cell_line(adata):
    KNOWN = ["k562", "rpe1", "hesc", "h1", "jurkat", "ipsc", "hela", "a549",
             "293t", "hek293", "thp1", "mcf7", "u2os", "hct116"]
    haystack = str(adata.uns).lower()
    for c in adata.obs.columns:
        haystack += str(adata.obs[c].iloc[0]).lower() + " "
    for cl in KNOWN:
        if cl in haystack:
            return cl.upper()
    return "UNKNOWN"


def _get_pert_type(adata):
    haystack = str(adata.uns).lower() + " ".join(adata.obs.columns).lower()
    if "crispri" in haystack or "dcas9" in haystack or "krab" in haystack:
        return "CRISPRi"
    if "crispra" in haystack:
        return "CRISPRa"
    if "crispr" in haystack or "cas9" in haystack or "sgrna" in haystack:
        return "CRISPR"
    return "CRISPRi"


# ── Main ──────────────────────────────────────────────────────────────────────

def detect(h5ad_path):
    path = Path(h5ad_path)
    if not path.exists():
        sys.exit(f"ERROR: File not found → {h5ad_path}")

    print(f"\n{'═'*70}")
    print(f"  Config Detective")
    print(f"  File: {path.name}")
    print(f"{'═'*70}\n")

    print("Loading metadata (backed mode)...")
    try:
        adata = sc.read_h5ad(h5ad_path, backed='r')
    except Exception as e:
        sys.exit(f"ERROR reading file: {e}")

    obs_cols = list(adata.obs.columns)
    print(f"  Shape : {adata.shape[0]:,} cells × {adata.shape[1]:,} genes\n")

    # ── 1. All obs columns ────────────────────────────────────────────────
    print(SEP)
    print("  1 · ALL .obs METADATA COLUMNS")
    print(SEP)
    for i, c in enumerate(obs_cols):
        dtype  = str(adata.obs[c].dtype)
        n_uniq = adata.obs[c].nunique()
        sample = adata.obs[c].dropna().astype(str).unique()[:4].tolist()
        print(f"  [{i:02d}]  {c:<40}  unique={n_uniq:<8}  e.g. {sample}")
    print()

    # ── 2. uns keys ───────────────────────────────────────────────────────
    print(SEP)
    print("  2 · UNS KEYS")
    print(SEP)
    if adata.uns:
        for k, v in adata.uns.items():
            print(f"  {k}: {str(v)[:120]}")
    else:
        print("  (empty)")
    print()

    # ── 3. Perturbation column candidates ────────────────────────────────
    print(SEP)
    print("  3 · PERTURBATION COLUMN CANDIDATES")
    print(SEP)

    best_pert_col    = None
    best_ctrl_label  = None
    best_ctrl_method = "not_found"

    pert_candidates = _top_pert_candidates(obs_cols, {}, n=6)

    for col, score in pert_candidates:
        n_uniq   = adata.obs[col].nunique()
        examples = adata.obs[col].dropna().astype(str).unique()[:6].tolist()
        looks_ok, reason = _looks_like_pert_column(adata.obs[col])
        ctrl_lbl, method = _find_control_label(adata.obs[col])

        if best_pert_col is None and looks_ok:
            marker = "  ★ BEST MATCH"
        elif not looks_ok:
            marker = "  ✗ REJECTED  "
        else:
            marker = "               "

        print(f"{marker}  '{col}'  (score={score})")
        print(f"           unique={n_uniq:,}   e.g. {examples}")
        if not looks_ok:
            print(f"           reason: {reason}")
        if ctrl_lbl:
            print(f"           control: '{ctrl_lbl}'  [{method}]")
        else:
            print(f"           control: not found")
        print()

        if best_pert_col is None and looks_ok:
            best_pert_col    = col
            best_ctrl_label  = ctrl_lbl
            best_ctrl_method = method

    # Fallback: if all vocabulary candidates failed, pick by unique-value count
    if best_pert_col is None:
        print("  ⚠  All vocabulary candidates failed. Trying fallback...\n")
        str_cols = [(c, adata.obs[c].nunique()) for c in obs_cols
                    if str(adata.obs[c].dtype) in ('object', 'category')]
        str_cols.sort(key=lambda x: -x[1])
        for col, n_uniq in str_cols[:5]:
            looks_ok, _ = _looks_like_pert_column(adata.obs[col])
            if looks_ok:
                ctrl_lbl, method = _find_control_label(adata.obs[col])
                examples = adata.obs[col].dropna().astype(str).unique()[:4].tolist()
                print(f"  → Fallback selected: '{col}'  unique={n_uniq}  e.g. {examples}")
                best_pert_col    = col
                best_ctrl_label  = ctrl_lbl
                best_ctrl_method = method
                break
        print()

    # ── 4. Full value counts for best perturbation column ────────────────
    print(SEP)
    print(f"  4 · VALUE COUNTS — '{best_pert_col}'")
    print(SEP)
    if best_pert_col:
        counts = Counter(adata.obs[best_pert_col].dropna().astype(str))
        total  = sum(counts.values())
        print(f"  {'Label':<45}  {'Cells':>8}  {'%':>6}")
        print(f"  {'─'*45}  {'─'*8}  {'─'*6}")
        for lbl, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            pct = cnt / total * 100
            # Flag exact-match controls only (not substring to avoid false flags)
            is_ctrl = any(_col_lower(lbl) == _col_lower(v)
                          for v in CTRL_VOCAB_ORDERED[:10])
            flag = "  ← CONTROL" if is_ctrl else ""
            print(f"  {lbl:<45}  {cnt:>8,}  {pct:>5.1f}%{flag}")
    print()

    # ── 5. Batch column ───────────────────────────────────────────────────
    print(SEP)
    print("  5 · BATCH / REPLICATE COLUMN CANDIDATES")
    print(SEP)
    batch_candidates = _top_batch_candidates(obs_cols, n=4)
    best_batch_col   = None
    if batch_candidates:
        for col, score in batch_candidates:
            n_uniq   = adata.obs[col].nunique()
            examples = adata.obs[col].dropna().astype(str).unique()[:5].tolist()
            marker   = "  ★ BEST MATCH" if best_batch_col is None else "               "
            print(f"{marker}  '{col}'  (score={score})")
            print(f"               unique={n_uniq}   e.g. {examples}\n")
            if best_batch_col is None:
                best_batch_col = col
    else:
        print("  No batch column found — likely single-batch.\n")

    # ── 6. Organism / cell line ───────────────────────────────────────────
    print(SEP)
    print("  6 · ORGANISM & CELL LINE")
    print(SEP)
    organism, org_evidence = _detect_organism(adata)
    cell_line  = _detect_cell_line(adata)
    pert_type  = _get_pert_type(adata)
    dset_name  = path.stem.replace(" ", "_")
    print(f"  Organism  : {organism}  [{org_evidence}]")
    print(f"  Cell line : {cell_line}")
    print(f"  Pert type : {pert_type}\n")

    # ── 7. Var columns ───────────────────────────────────────────────────
    print(SEP)
    print("  7 · VAR COLUMNS")
    print(SEP)
    for c in adata.var.columns:
        sample = adata.var[c].dropna().iloc[:3].tolist() if len(adata.var) > 0 else []
        print(f"  {c:<40}  e.g. {sample}")
    print(f"\n  First 10 var_names: {list(adata.var_names[:10])}\n")

    # ── 8. YAML suggestion ───────────────────────────────────────────────
    ctrl_yaml  = best_ctrl_label  or "UNKNOWN_CONTROL"
    batch_yaml = f'"{best_batch_col}"' if best_batch_col else "null"

    print(f"{'═'*70}")
    print(f"  SUGGESTED CONFIG BLOCK")
    print(f"{'═'*70}")
    print(f"""
dataset:
  perturbation_col:  "{best_pert_col or 'UNKNOWN'}"
  control_label:     "{ctrl_yaml}"    # method: {best_ctrl_method}
  batch_col:         {batch_yaml}

  # Also useful for SPORE:
  name:              "{dset_name}"
  organism:          "{organism}"     # evidence: {org_evidence}
  cell_line:         "{cell_line}"    # ⚠ verify
  perturbation_type: "{pert_type}"    # ⚠ verify
""")

    # ── 9. Confidence summary ────────────────────────────────────────────
    print(SEP)
    print("  CONFIDENCE SUMMARY")
    print(SEP)
    checks = [
        ("perturbation_col",  best_pert_col  is not None, best_pert_col),
        ("control_label",     best_ctrl_label is not None, f"{ctrl_yaml}  [{best_ctrl_method}]"),
        ("batch_col",         best_batch_col is not None, best_batch_col or "null"),
        ("organism",          org_evidence != "default_fallback", f"{organism}  [{org_evidence}]"),
    ]
    all_ok = True
    for field, ok, value in checks:
        icon = "✓" if ok else "⚠"
        if not ok:
            all_ok = False
        print(f"  {icon}  {field:<20}  {value}")

    if all_ok:
        print("\n  All fields detected. Review YAML block above before using.")
    else:
        print("\n  ⚠ One or more fields need manual review.")
    print()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else H5AD_PATH
    detect(path)
