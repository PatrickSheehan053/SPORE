"""
SPORE · src/plotting.py
───────────────────────
Publication-quality plotting utilities for every SPORE phase.
All scatter/violin plots DOWNSAMPLE to avoid OOM on 2M+ cell datasets.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

# Maximum points to render in scatter/violin plots
_MAX_PLOT_POINTS = 200_000


# ── Theme Setup ─────────────────────────────────────────────────────────────

_DARK = {
    "bg": "#0D1117", "panel": "#161B22", "text": "#E6EDF3",
    "grid": "#21262D", "accent": "#58A6FF", "warn": "#F85149",
    "good": "#3FB950", "muted": "#8B949E", "highlight": "#D2A8FF",
}

_LIGHT = {
    "bg": "#FFFFFF", "panel": "#F6F8FA", "text": "#1F2328",
    "grid": "#D1D9E0", "accent": "#0969DA", "warn": "#CF222E",
    "good": "#1A7F37", "muted": "#656D76", "highlight": "#8250DF",
}


def _get_theme(cfg: dict) -> dict:
    return _DARK if cfg["plotting"]["style"] == "dark" else _LIGHT


def apply_spore_style(cfg: dict):
    """Apply the SPORE matplotlib style globally."""
    theme = _get_theme(cfg)
    plt.rcParams.update({
        "figure.facecolor": theme["bg"],
        "axes.facecolor": theme["panel"],
        "axes.edgecolor": theme["grid"],
        "axes.labelcolor": theme["text"],
        "text.color": theme["text"],
        "xtick.color": theme["text"],
        "ytick.color": theme["text"],
        "grid.color": theme["grid"],
        "grid.alpha": 0.5,
        "figure.dpi": cfg["plotting"]["dpi"],
        "font.family": "monospace",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "legend.facecolor": theme["panel"],
        "legend.edgecolor": theme["grid"],
        "savefig.facecolor": theme["bg"],
        "savefig.bbox": "tight",
        "savefig.dpi": cfg["plotting"]["dpi"],
    })


def _save(fig, cfg: dict, filename: str):
    if cfg["plotting"]["save_figures"]:
        fmt = cfg["plotting"]["figure_format"]
        path = cfg["paths"]["_figures"] / f"{filename}.{fmt}"
        fig.savefig(path, facecolor=fig.get_facecolor())
        return path
    return None


def _format_ax(ax, theme: dict, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_title(title, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(theme["grid"])


def _downsample_series(s: pd.Series, max_n: int = _MAX_PLOT_POINTS) -> np.ndarray:
    """Downsample a pandas Series to at most max_n values for plotting."""
    vals = s.dropna().values
    if len(vals) > max_n:
        rng = np.random.default_rng(42)
        vals = rng.choice(vals, size=max_n, replace=False)
    return vals


# ── Phase 1 Plots ──────────────────────────────────────────────────────────

def _ensure_qc_metrics(adata):
    """Silently calculate QC metrics if they are missing from a raw matrix before plotting."""
    if "total_counts" not in adata.obs.columns or "pct_counts_mt" not in adata.obs.columns:
        import scanpy as sc
        adata.var['mt'] = adata.var_names.str.startswith('MT-') | adata.var_names.str.startswith('mt-')
        sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

def plot_qc_violin(adata, cfg: dict, phase_label: str = "pre"):
    """Phase 1: Violin plots of QC metrics — downsampled for large datasets."""
    _ensure_qc_metrics(adata)  # <-- FIREWALL
    
    theme = _get_theme(cfg)
    metrics = []
    titles = []

    if "n_genes_by_counts" in adata.obs.columns:
        metrics.append("n_genes_by_counts")
        titles.append("Genes per Cell")
    if "total_counts" in adata.obs.columns:
        metrics.append("total_counts")
        titles.append("Total UMI Counts")
    if "pct_counts_mt" in adata.obs.columns:
        metrics.append("pct_counts_mt")
        titles.append("% Mitochondrial")

    n = len(metrics)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, metric, title in zip(axes, metrics, titles):
        data = _downsample_series(adata.obs[metric])
        vp = ax.violinplot(data, positions=[0], showmedians=True, showextrema=False)
        for body in vp["bodies"]:
            body.set_facecolor(theme["accent"])
            body.set_alpha(0.6)
        vp["cmedians"].set_color(theme["highlight"])
        vp["cmedians"].set_linewidth(2)

        if metric == "pct_counts_mt":
            thresh = cfg["phase1_cell_triage"]["mt_threshold"] * 100
            ax.axhline(thresh, color=theme["warn"], ls="--", lw=1.5,
                       label=f"Threshold: {thresh:.0f}%")
            ax.legend(fontsize=9, loc="upper right")
        if metric == "n_genes_by_counts":
            lo = cfg["phase1_cell_triage"]["min_genes_per_cell"]
            hi = cfg["phase1_cell_triage"]["max_genes_per_cell"]
            ax.axhline(lo, color=theme["warn"], ls="--", lw=1, label=f"Min: {lo}")
            ax.axhline(hi, color=theme["warn"], ls="--", lw=1, label=f"Max: {hi:,}")
            ax.legend(fontsize=9, loc="upper right")

        ax.set_xticks([])
        _format_ax(ax, theme, title, ylabel=metric.replace("_", " ").title())
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{x:,.0f}"))

    total_cells = adata.n_obs
    tag = "Pre-Filter" if phase_label == "pre" else "Post-Filter"
    sample_note = f" (sampled {_MAX_PLOT_POINTS:,})" if total_cells > _MAX_PLOT_POINTS else ""
    fig.suptitle(f"SPORE Phase 1 · Cell QC ({tag}, n={total_cells:,}{sample_note})",
                 fontsize=16, fontweight="bold", color=theme["text"], y=1.02)
    fig.tight_layout()
    _save(fig, cfg, f"phase1_qc_violin_{phase_label}")
    plt.show()


def plot_mt_scatter(adata, cfg: dict):
    """Phase 1: Scatter of total counts vs MT fraction — downsampled."""
    _ensure_qc_metrics(adata)  # <-- FIREWALL
    
    theme = _get_theme(cfg)
    thresh = cfg["phase1_cell_triage"]["mt_threshold"] * 100
    total_cells = adata.n_obs

    # Downsample
    if total_cells > _MAX_PLOT_POINTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(total_cells, size=_MAX_PLOT_POINTS, replace=False)
        tc = adata.obs["total_counts"].values[idx]
        mt = adata.obs["pct_counts_mt"].values[idx]
    else:
        tc = adata.obs["total_counts"].values
        mt = adata.obs["pct_counts_mt"].values

    fig, ax = plt.subplots(figsize=(8, 5))
    mask = mt <= thresh
    colors = np.where(mask, theme["good"], theme["warn"])
    ax.scatter(tc, mt, c=colors, s=1, alpha=0.3, rasterized=True)
    ax.axhline(thresh, color=theme["warn"], ls="--", lw=1.5,
               label=f"MT threshold: {thresh:.0f}%")

    # Compute pass/fail on FULL data (not sample)
    full_pass = (adata.obs["pct_counts_mt"] <= thresh).sum()
    full_fail = total_cells - full_pass
    ax.legend(fontsize=10, loc="upper right")
    ax.text(0.02, 0.95, f"Pass: {full_pass:,}  |  Fail: {full_fail:,}",
            transform=ax.transAxes, fontsize=10, color=theme["text"],
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", fc=theme["panel"],
                      ec=theme["grid"], alpha=0.9))

    sample_note = f" (sampled {_MAX_PLOT_POINTS:,})" if total_cells > _MAX_PLOT_POINTS else ""
    _format_ax(ax, theme, f"SPORE Phase 1 · MT% vs Total Counts{sample_note}",
               xlabel="Total UMI Counts", ylabel="% Mitochondrial")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k"))
    fig.tight_layout()
    _save(fig, cfg, "phase1_mt_scatter")
    plt.show()


def plot_filtering_waterfall(counts: dict, cfg: dict, phase: str = "phase1"):
    """Waterfall chart showing cell/gene counts after each filter step."""
    theme = _get_theme(cfg)
    steps = list(counts.keys())
    values = list(counts.values())

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(range(len(steps)), values, color=theme["accent"], height=0.6,
                   edgecolor=theme["grid"], linewidth=0.5)
    bars[0].set_facecolor(theme["muted"])

    for i, (step, val) in enumerate(zip(steps, values)):
        pct = val / values[0] * 100 if values[0] > 0 else 0
        ax.text(val + values[0] * 0.01, i,
                f" {val:,}  ({pct:.1f}%)", va="center",
                fontsize=10, color=theme["text"])

    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels(steps, fontsize=10)
    ax.invert_yaxis()
    _format_ax(ax, theme, f"SPORE {phase.replace('_', ' ').title()} · Filtering Waterfall",
               xlabel="Count")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))
    fig.tight_layout()
    _save(fig, cfg, f"{phase}_waterfall")
    plt.show()


# ── Phase 2 Plots ──────────────────────────────────────────────────────────

def plot_escaper_summary(escaper_stats: pd.DataFrame, cfg: dict):
    """Phase 2: Bar chart of per-perturbation escaper rates (top 40)."""
    theme = _get_theme(cfg)
    df = escaper_stats.sort_values("n_escaped", ascending=False).head(40)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(df))
    ax.bar(x, df["n_total"], color=theme["muted"], alpha=0.5, label="Total", width=0.7)
    ax.bar(x, df["n_kept"], color=theme["good"], alpha=0.8, label="Kept", width=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(df["perturbation"], rotation=90, fontsize=7)
    ax.legend(fontsize=10)
    _format_ax(ax, theme, "SPORE Phase 2 · Escaper Filtering (Top 40 Perturbations)",
               ylabel="Cell Count")
    fig.tight_layout()
    _save(fig, cfg, "phase2_escaper_summary")
    plt.show()


def plot_perturbation_sizes(sizes: pd.Series, cfg: dict, min_thresh: int = 50):
    """Phase 2: Histogram of cells-per-perturbation with min threshold."""
    theme = _get_theme(cfg)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(sizes.values, bins=80, color=theme["accent"], alpha=0.7,
            edgecolor=theme["panel"], linewidth=0.5)
    ax.axvline(min_thresh, color=theme["warn"], ls="--", lw=2,
               label=f"Min threshold: {min_thresh}")

    n_below = (sizes < min_thresh).sum()
    ax.text(0.97, 0.95,
            f"Below threshold: {n_below} perturbations",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, color=theme["warn"],
            bbox=dict(boxstyle="round,pad=0.3", fc=theme["panel"],
                      ec=theme["warn"], alpha=0.9))

    ax.legend(fontsize=10)
    _format_ax(ax, theme, "SPORE Phase 2 · Cells per Perturbation",
               xlabel="Number of Cells", ylabel="Number of Perturbations")
    fig.tight_layout()
    _save(fig, cfg, "phase2_perturbation_sizes")
    plt.show()


# ── Phase 3 Plots ──────────────────────────────────────────────────────────

def plot_gene_expression_hist(adata, cfg: dict, phase_label: str = "pre"):
    """Phase 3: Histogram of mean gene expression across cells."""
    theme = _get_theme(cfg)
    import scipy.sparse as sp

    X = adata.X
    if sp.issparse(X):
        # SAFE: .sum() avoids the intermediate dense allocations of .mean()
        mean_expr = np.ravel(X.sum(axis=0)) / adata.n_obs
    else:
        mean_expr = X.mean(axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.log1p(mean_expr), bins=100, color=theme["accent"], alpha=0.7,
            edgecolor=theme["panel"], linewidth=0.3)
    _format_ax(ax, theme,
               f"SPORE Phase 3 · Gene Mean Expression ({phase_label.title()})",
               xlabel="log1p(Mean UMI)", ylabel="Number of Genes")
    fig.tight_layout()
    _save(fig, cfg, f"phase3_gene_expr_hist_{phase_label}")
    plt.show()


def plot_rescued_genes(rescued: list, n_hvg: int, cfg: dict):
    """Phase 3/5: Summary of rescued perturbation targets."""
    theme = _get_theme(cfg)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.barh(["HVG Selected", "Rescued Targets"],
            [n_hvg - len(rescued), len(rescued)],
            color=[theme["accent"], theme["highlight"]],
            edgecolor=theme["grid"], height=0.5)
    for i, v in enumerate([n_hvg - len(rescued), len(rescued)]):
        ax.text(v + 20, i, f"{v:,}", va="center", fontsize=11, color=theme["text"])
    _format_ax(ax, theme, "SPORE · Perturbation Target Rescue", xlabel="Gene Count")
    fig.tight_layout()
    _save(fig, cfg, "rescued_targets")
    plt.show()


# ── Phase 4 Plots ──────────────────────────────────────────────────────────

def plot_split_summary(split_info: dict, cfg: dict):
    """Phase 4: Bar + pie showing train/val/test proportions."""
    theme = _get_theme(cfg)
    labels = list(split_info.keys())
    values = list(split_info.values())
    total = sum(values)
    colors = [theme["accent"], theme["highlight"], theme["warn"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5),
                                    gridspec_kw={"width_ratios": [1.5, 1]})

    bars = ax1.bar(labels, values, color=colors[:len(labels)], width=0.5,
                   edgecolor=theme["grid"], linewidth=0.5)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.01,
                 f"{val:,}\n({val/total*100:.1f}%)", ha="center", va="bottom",
                 fontsize=10, color=theme["text"])
    _format_ax(ax1, theme, "Split Counts", ylabel="Count")

    wedges, texts, autotexts = ax2.pie(
        values, labels=labels, colors=colors[:len(labels)],
        autopct="%1.1f%%", startangle=90,
        textprops={"color": theme["text"], "fontsize": 11})
    for t in autotexts:
        t.set_fontsize(10)
        t.set_color(theme["bg"])
        t.set_fontweight("bold")
    ax2.set_title("Split Proportions", fontsize=14, fontweight="bold",
                  color=theme["text"], pad=12)

    fig.suptitle("SPORE Phase 4 · Data Split Summary",
                 fontsize=16, fontweight="bold", color=theme["text"], y=1.02)
    fig.tight_layout()
    _save(fig, cfg, "phase4_split_summary")
    plt.show()


def plot_deg_stratification(deg_counts: pd.Series, bins: int, cfg: dict):
    """Phase 4a: Histogram of DEG counts with stratification bins."""
    theme = _get_theme(cfg)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(deg_counts.values, bins=50, color=theme["accent"], alpha=0.7,
            edgecolor=theme["panel"])

    quantiles = np.linspace(0, 1, bins + 1)[1:-1]
    for q in quantiles:
        val = np.quantile(deg_counts.values, q)
        ax.axvline(val, color=theme["highlight"], ls="--", lw=1.5, alpha=0.8)

    _format_ax(ax, theme, "SPORE Phase 4a · DEG Distribution for Stratified Splitting",
               xlabel="Number of DEGs per Perturbation", ylabel="Count")
    fig.tight_layout()
    _save(fig, cfg, "phase4a_deg_stratification")
    plt.show()


# ── Phase 5 Plots ──────────────────────────────────────────────────────────

def plot_hvg_variance(adata, cfg: dict):
    """Phase 5: Scatter of mean vs dispersion/variance, highlighting HVGs."""
    theme = _get_theme(cfg)
    
    if "hvg_stats" in adata.uns:
        var_df = adata.uns["hvg_stats"]
    else:
        var_df = adata.var

    if "highly_variable" not in var_df.columns:
        return

    means = var_df.get("means")
    # seurat flavor uses dispersions_norm, seurat_v3 uses variances_norm
    dispersions = var_df.get("dispersions_norm")
    if dispersions is None:
        dispersions = var_df.get("variances_norm")

    if means is None or dispersions is None:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    hv = var_df["highly_variable"]

    ax.scatter(means[~hv], dispersions[~hv], s=2, alpha=0.3,
               c=theme["muted"], label="Non-HVG", rasterized=True)
    ax.scatter(means[hv], dispersions[hv], s=4, alpha=0.5,
               c=theme["accent"], label=f"HVG (n={hv.sum():,})", rasterized=True)
    ax.legend(fontsize=10, markerscale=4)
    _format_ax(ax, theme, "SPORE Phase 5 · Highly Variable Gene Selection",
               xlabel="Mean Expression", ylabel="Normalized Variance / Dispersion")
    fig.tight_layout()
    _save(fig, cfg, "phase5_hvg_scatter")
    plt.show()


# ── Phase 6/7 Plots ────────────────────────────────────────────────────────

def plot_normalization_comparison(raw_counts, norm_counts, cfg: dict):
    """Phase 6: Before/after histograms of expression values."""
    theme = _get_theme(cfg)
    import scipy.sparse as sp
    
    # SAFE: Extract ONLY the non-zero C-buffer values directly.
    # Passing a sparse matrix directly to plt.hist() triggers np.asarray() and OOMs.
    if sp.issparse(raw_counts):
        raw_vals = raw_counts.data
    else:
        raw_vals = np.ravel(raw_counts)
        raw_vals = raw_vals[raw_vals > 0] # Filter 0s for fair comparison

    if sp.issparse(norm_counts):
        norm_vals = norm_counts.data
    else:
        norm_vals = np.ravel(norm_counts)
        norm_vals = norm_vals[norm_vals > 0]

    # Downsample massive arrays to avoid Matplotlib CPU hang
    if len(raw_vals) > _MAX_PLOT_POINTS:
        rng = np.random.default_rng(42)
        raw_vals = rng.choice(raw_vals, size=_MAX_PLOT_POINTS, replace=False)
    if len(norm_vals) > _MAX_PLOT_POINTS:
        rng = np.random.default_rng(42)
        norm_vals = rng.choice(norm_vals, size=_MAX_PLOT_POINTS, replace=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.hist(raw_vals, bins=100, color=theme["muted"], alpha=0.7)
    _format_ax(ax1, theme, "Raw Counts (Non-Zero)", xlabel="UMI Count", ylabel="Frequency")

    ax2.hist(norm_vals, bins=100, color=theme["good"], alpha=0.7)
    _format_ax(ax2, theme, "After log1p Normalization",
               xlabel="log1p(Normalized)", ylabel="Frequency")

    fig.suptitle("SPORE Phase 6 · Normalization Effect",
                 fontsize=16, fontweight="bold", color=theme["text"], y=1.02)
    fig.tight_layout()
    _save(fig, cfg, "phase6_normalization")
    plt.show()


def plot_cell_cycle_scores(adata, cfg: dict):
    """Phase 7: Scatter of S score vs G2M score — downsampled."""
    theme = _get_theme(cfg)
    if "S_score" not in adata.obs.columns:
        return

    n = adata.n_obs
    if n > _MAX_PLOT_POINTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=_MAX_PLOT_POINTS, replace=False)
        obs_sub = adata.obs.iloc[idx]
    else:
        obs_sub = adata.obs

    fig, ax = plt.subplots(figsize=(7, 7))
    phase_colors = {"G1": theme["muted"], "S": theme["accent"], "G2M": theme["highlight"]}

    for phase, color in phase_colors.items():
        mask = obs_sub["phase"] == phase
        ax.scatter(obs_sub.loc[mask, "S_score"],
                   obs_sub.loc[mask, "G2M_score"],
                   c=color, s=2, alpha=0.3, label=phase, rasterized=True)

    ax.legend(fontsize=10, markerscale=5)
    _format_ax(ax, theme, "SPORE Phase 7 · Cell Cycle Phase Assignment",
               xlabel="S Score", ylabel="G2M Score")
    ax.set_aspect("equal")
    fig.tight_layout()
    _save(fig, cfg, "phase7_cell_cycle")
    plt.show()


def plot_batch_umap(adata, cfg: dict, label: str = "pre"):
    """Phase 7: UMAP colored by batch — downsampled."""
    theme = _get_theme(cfg)
    batch_col = cfg["dataset"]["batch_col"]
    if batch_col is None or "X_umap" not in adata.obsm:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    umap = adata.obsm["X_umap"]
    batches = adata.obs[batch_col].astype("category")
    n_batches = batches.nunique()

    # Downsample
    n = adata.n_obs
    if n > _MAX_PLOT_POINTS:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=_MAX_PLOT_POINTS, replace=False)
    else:
        idx = np.arange(n)

    cmap = plt.cm.get_cmap("tab20", n_batches)
    for i, batch in enumerate(batches.cat.categories):
        mask = np.where(batches.values[idx] == batch)[0]
        ax.scatter(umap[idx[mask], 0], umap[idx[mask], 1],
                   c=[cmap(i)], s=1, alpha=0.2, label=batch, rasterized=True)

    if n_batches <= 15:
        ax.legend(fontsize=7, markerscale=5, ncol=2, loc="lower right")

    tag = "Before Correction" if label == "pre" else "After Correction"
    _format_ax(ax, theme, f"SPORE Phase 7 · Batch Distribution ({tag})",
               xlabel="UMAP 1", ylabel="UMAP 2")
    fig.tight_layout()
    _save(fig, cfg, f"phase7_batch_umap_{label}")
    plt.show()


# ── Pipeline Summary ───────────────────────────────────────────────────────

def plot_pipeline_summary(phase_counts: dict, cfg: dict):
    """Final summary: horizontal bar chart of cells remaining at each phase."""
    theme = _get_theme(cfg)
    phases = list(phase_counts.keys())
    counts = list(phase_counts.values())
    start = counts[0]

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = []
    for c in counts:
        pct = c / start
        if pct > 0.8:
            colors.append(theme["good"])
        elif pct > 0.5:
            colors.append(theme["accent"])
        elif pct > 0.3:
            colors.append(theme["highlight"])
        else:
            colors.append(theme["warn"])

    bars = ax.barh(range(len(phases)), counts, color=colors, height=0.6,
                   edgecolor=theme["grid"], linewidth=0.5)

    for i, (phase, cnt) in enumerate(zip(phases, counts)):
        pct = cnt / start * 100
        ax.text(cnt + start * 0.005, i,
                f" {cnt:,}  ({pct:.1f}%)", va="center",
                fontsize=10, color=theme["text"])

    ax.set_yticks(range(len(phases)))
    ax.set_yticklabels(phases, fontsize=11)
    ax.invert_yaxis()
    _format_ax(ax, theme, "SPORE · Pipeline Summary", xlabel="Cell Count")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{x/1000:.0f}k" if x >= 1000 else f"{x:.0f}"))
    fig.tight_layout()
    _save(fig, cfg, "pipeline_summary")
    plt.show()