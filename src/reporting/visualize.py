"""
Publication-Quality Visualisations for the Domain Adaptation Benchmark.

Generates matplotlib figures suitable for research papers, presentations,
and MS admission portfolios.  All figures use a consistent dark theme
with colour-blind-friendly palettes.

Available plots
---------------
    1. accuracy_heatmap()       — 15x4 corruption x method heatmap
    2. mce_bar_chart()          — mCE comparison across methods
    3. entropy_gain_scatter()   — RQ3: entropy vs. TENT gain with regression
    4. category_comparison()    — grouped bars by corruption category
    5. pl_failure_chart()       — pseudo-label degradation on blur
    6. severity_analysis()      — accuracy vs. severity level
    7. generate_all_figures()   — convenience: generate all plots to a directory

Usage
-----
    from src.reporting.visualize import generate_all_figures
    generate_all_figures(summary, uncertainty_analyzer, output_dir="./figures")
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Lazy import matplotlib to avoid import errors in CI environments
_MPL_AVAILABLE = False
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend for server/notebook
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    _MPL_AVAILABLE = True
except ImportError:
    logger.warning(
        "matplotlib not available. Install with: pip install matplotlib"
    )

from src.models import (
    CORRUPTION_CATEGORIES,
    METHOD_DISPLAY,
    BLUR_CORRUPTIONS,
    BenchmarkSummary,
)

# ── Style configuration ─────────────────────────────────────────────────────

# Research-friendly colour palette (colour-blind safe)
COLOURS = {
    "no_adaptation":  "#6C757D",   # grey
    "test_time_norm": "#0D6EFD",   # blue
    "tent":           "#198754",   # green
    "pseudo_label":   "#DC3545",   # red
}

CATEGORY_COLOURS = {
    "noise":   "#FF6B6B",
    "blur":    "#4ECDC4",
    "weather": "#45B7D1",
    "digital": "#96CEB4",
}

_METHOD_ORDER = ["no_adaptation", "test_time_norm", "tent", "pseudo_label"]


def _setup_style():
    """Apply publication-quality dark theme."""
    if not _MPL_AVAILABLE:
        return
    plt.rcParams.update({
        "figure.facecolor":   "#0d0d18",
        "axes.facecolor":     "#14142a",
        "axes.edgecolor":     "#2a2a55",
        "axes.labelcolor":    "#c8c8e8",
        "text.color":         "#c8c8e8",
        "xtick.color":        "#7878a8",
        "ytick.color":        "#7878a8",
        "grid.color":         "#2a2a55",
        "grid.alpha":         0.5,
        "font.family":        "sans-serif",
        "font.size":          11,
        "axes.titlesize":     14,
        "axes.labelsize":     12,
        "legend.facecolor":   "#1c1c35",
        "legend.edgecolor":   "#2a2a55",
        "legend.fontsize":    10,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "#0d0d18",
    })


# ── Plot functions ───────────────────────────────────────────────────────────


def accuracy_heatmap(
    summary: BenchmarkSummary,
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Generate a corruption x method accuracy heatmap.

    Parameters
    ----------
    summary : BenchmarkSummary
        Benchmark results.
    save_path : str, optional
        Path to save the figure. If None, returns the figure object.

    Returns
    -------
    matplotlib.figure.Figure or None
    """
    if not _MPL_AVAILABLE:
        logger.warning("matplotlib not available, skipping heatmap.")
        return None

    _setup_style()

    corruptions = summary.corruption_types
    methods = _METHOD_ORDER
    method_labels = [METHOD_DISPLAY.get(m, m) for m in methods]

    # Build accuracy matrix
    data = np.zeros((len(corruptions), len(methods)))
    for i, c in enumerate(corruptions):
        for j, m in enumerate(methods):
            acc = summary.accuracy_table.get(c, {}).get(m, 0.0)
            data[i, j] = acc * 100  # Convert to percentage

    fig, ax = plt.subplots(figsize=(10, max(6, len(corruptions) * 0.45)))

    # Custom colourmap: red -> yellow -> green
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "accuracy", ["#6b1a1a", "#5a6b1a", "#1a6b3a"], N=256
    )

    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=100)

    # Axes
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(method_labels, fontweight="bold")
    ax.set_yticks(range(len(corruptions)))
    ax.set_yticklabels(corruptions, fontfamily="monospace", fontsize=9)

    # Annotate cells with accuracy values
    for i in range(len(corruptions)):
        for j in range(len(methods)):
            val = data[i, j]
            text_colour = "white" if val < 65 else "#0d0d18"
            winner = summary.winners.get(corruptions[i], "")
            weight = "bold" if methods[j] == winner else "normal"
            ax.text(
                j, i, f"{val:.1f}%",
                ha="center", va="center",
                color=text_colour, fontsize=8, fontweight=weight,
            )

    ax.set_title(
        f"Accuracy Heatmap — Severity {summary.severity}",
        fontweight="bold", pad=15,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Accuracy (%)", color="#c8c8e8")
    cbar.ax.tick_params(colors="#7878a8")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info("Saved accuracy heatmap -> %s", save_path)
        plt.close(fig)
        return None
    return fig


def mce_bar_chart(
    summary: BenchmarkSummary,
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Generate a bar chart comparing mCE across methods.
    """
    if not _MPL_AVAILABLE:
        return None

    _setup_style()

    methods = [m for m in _METHOD_ORDER if m in summary.mce_scores]
    mce_vals = [summary.mce_scores[m] for m in methods]
    rel_imps = [summary.relative_improvements.get(m, 0.0) for m in methods]
    labels = [METHOD_DISPLAY.get(m, m) for m in methods]
    colours = [COLOURS.get(m, "#888888") for m in methods]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, mce_vals, color=colours, edgecolor="#2a2a55",
                  linewidth=1.5, width=0.6, zorder=3)

    # Annotate bars
    for bar, mce_val, rel in zip(bars, mce_vals, rel_imps):
        y = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, y + 0.005,
            f"{mce_val:.4f}\n({rel:+.1%})",
            ha="center", va="bottom", fontsize=9, fontweight="bold",
            color="#c8c8e8",
        )

    ax.set_ylabel("Mean Corruption Error (mCE) ↓", fontweight="bold")
    ax.set_title("mCE Comparison Across TTA Methods", fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.set_ylim(0, max(mce_vals) * 1.25)

    # Highlight best
    best_idx = mce_vals.index(min(mce_vals))
    bars[best_idx].set_edgecolor("#3adb7a")
    bars[best_idx].set_linewidth(3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info("Saved mCE bar chart -> %s", save_path)
        plt.close(fig)
        return None
    return fig


def entropy_gain_scatter(
    uncertainty_analyzer,
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Generate scatter plot: pre-adaptation entropy vs. TENT adaptation gain.

    Includes linear regression line and Pearson r annotation (RQ3).
    """
    if not _MPL_AVAILABLE:
        return None

    _setup_style()

    records = uncertainty_analyzer.get_records()
    if len(records) < 2:
        logger.warning("Need at least 2 corruption types for scatter plot.")
        return None

    corruptions = list(records.keys())
    entropies = [records[c]["mean_entropy"] for c in corruptions]
    gains = [records[c]["adaptation_gain"] for c in corruptions]

    # Determine category for colour coding
    cat_map = {}
    for cat, members in CORRUPTION_CATEGORIES.items():
        for m in members:
            cat_map[m] = cat

    fig, ax = plt.subplots(figsize=(9, 6))

    # Plot points by category
    for cat, colour in CATEGORY_COLOURS.items():
        cat_x = [e for c, e in zip(corruptions, entropies)
                 if cat_map.get(c) == cat]
        cat_y = [g for c, g in zip(corruptions, gains)
                 if cat_map.get(c) == cat]
        cat_labels = [c for c in corruptions if cat_map.get(c) == cat]
        if cat_x:
            ax.scatter(
                cat_x, cat_y, c=colour, s=100, alpha=0.85,
                edgecolors="white", linewidth=0.8,
                label=cat.capitalize(), zorder=5,
            )
            # Label each point
            for x, y, label in zip(cat_x, cat_y, cat_labels):
                short = label.replace("_", " ")
                if len(short) > 14:
                    short = short[:12] + "..."
                ax.annotate(
                    short, (x, y),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=7, color="#c8c8e8", alpha=0.8,
                )

    # Regression line
    if len(entropies) >= 2:
        z = np.polyfit(entropies, gains, 1)
        p = np.poly1d(z)
        x_line = np.linspace(
            min(entropies) * 0.95, max(entropies) * 1.05, 100
        )
        ax.plot(
            x_line, p(x_line), "--", color="#5a8fff", alpha=0.7,
            linewidth=2, label="Linear fit", zorder=4,
        )

    # Pearson r annotation
    r = uncertainty_analyzer.compute_correlation()
    ax.text(
        0.05, 0.95,
        f"Pearson r = {r:+.4f}",
        transform=ax.transAxes,
        fontsize=12, fontweight="bold",
        color="#3adb7a" if r > 0.4 else "#f0b429",
        va="top",
        bbox=dict(
            boxstyle="round,pad=0.5", facecolor="#1c1c35",
            edgecolor="#2a2a55",
        ),
    )

    ax.set_xlabel("Pre-Adaptation Entropy (nats)", fontweight="bold")
    ax.set_ylabel("TENT Adaptation Gain (Acc)", fontweight="bold")
    ax.set_title(
        "RQ3: Does Pre-Adaptation Entropy Predict TTA Benefit?",
        fontweight="bold", pad=15,
    )
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True, alpha=0.2, zorder=0)
    ax.axhline(y=0, color="#7878a8", linewidth=0.5, linestyle="-", alpha=0.5)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info("Saved entropy-gain scatter -> %s", save_path)
        plt.close(fig)
        return None
    return fig


def category_comparison(
    summary: BenchmarkSummary,
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Grouped bar chart: mean accuracy per corruption category per method.
    """
    if not _MPL_AVAILABLE:
        return None

    _setup_style()

    categories = list(CORRUPTION_CATEGORIES.keys())
    methods = [m for m in _METHOD_ORDER if m in summary.mce_scores]
    method_labels = [METHOD_DISPLAY.get(m, m) for m in methods]

    # Compute mean accuracy per category per method
    cat_data = {cat: {m: [] for m in methods} for cat in categories}
    for cat, members in CORRUPTION_CATEGORIES.items():
        for c in members:
            if c in summary.accuracy_table:
                for m in methods:
                    acc = summary.accuracy_table[c].get(m)
                    if acc is not None:
                        cat_data[cat][m].append(acc)

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.18
    offsets = np.arange(len(methods)) - (len(methods) - 1) / 2

    for i, method in enumerate(methods):
        means = []
        for cat in categories:
            vals = cat_data[cat][method]
            means.append(np.mean(vals) * 100 if vals else 0)
        bars = ax.bar(
            x + offsets[i] * width, means, width * 0.9,
            label=method_labels[i],
            color=COLOURS.get(method, "#888"),
            edgecolor="#2a2a55", linewidth=0.8,
            zorder=3,
        )
        # Annotate
        for bar, val in zip(bars, means):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}", ha="center", va="bottom",
                    fontsize=7, color="#c8c8e8",
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.capitalize() for c in categories], fontweight="bold",
    )
    ax.set_ylabel("Mean Accuracy (%)", fontweight="bold")
    ax.set_title(
        "Accuracy by Corruption Category — All Methods",
        fontweight="bold", pad=15,
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2, zorder=0)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info("Saved category comparison -> %s", save_path)
        plt.close(fig)
        return None
    return fig


def pl_failure_chart(
    summary: BenchmarkSummary,
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Side-by-side bar chart showing pseudo-label degradation on blur.
    """
    if not _MPL_AVAILABLE:
        return None

    _setup_style()

    failures = summary.pseudo_label_blur_failures
    if not failures:
        # Show all blur corruptions even if no failure detected
        blur_data = {}
        for c in BLUR_CORRUPTIONS:
            if c in summary.accuracy_table:
                base = summary.accuracy_table[c].get("no_adaptation", 0)
                pl = summary.accuracy_table[c].get("pseudo_label", 0)
                blur_data[c] = {"no_adaptation": base, "pseudo_label": pl}
        if not blur_data:
            logger.info(
                "No blur corruption data available for PL failure chart."
            )
            return None
        failures = blur_data

    corruptions = sorted(failures.keys())
    base_accs = [failures[c]["no_adaptation"] * 100 for c in corruptions]
    pl_accs = [failures[c]["pseudo_label"] * 100 for c in corruptions]

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(corruptions))
    width = 0.35

    ax.bar(
        x - width / 2, base_accs, width,
        label="No Adaptation (Baseline)",
        color=COLOURS["no_adaptation"], edgecolor="#2a2a55",
        linewidth=1.2, zorder=3,
    )
    ax.bar(
        x + width / 2, pl_accs, width,
        label="Pseudo-Label",
        color=COLOURS["pseudo_label"], edgecolor="#2a2a55",
        linewidth=1.2, zorder=3,
    )

    # Annotate degradation
    for i, c in enumerate(corruptions):
        delta = pl_accs[i] - base_accs[i]
        if delta < 0:
            ax.annotate(
                f"{delta:+.1f}%",
                xy=(x[i] + width / 2, pl_accs[i]),
                xytext=(0, -20), textcoords="offset points",
                ha="center", fontsize=9, fontweight="bold",
                color="#DC3545",
                arrowprops=dict(arrowstyle="->", color="#DC3545", lw=1.5),
            )

    labels = [c.replace("_", "\n") for c in corruptions]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Accuracy (%)", fontweight="bold")
    ax.set_title(
        "Pseudo-Label Confirmation Bias on Blur Corruptions",
        fontweight="bold", pad=15, color="#DC3545",
    )
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2, zorder=0)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info("Saved PL failure chart -> %s", save_path)
        plt.close(fig)
        return None
    return fig


def severity_analysis(
    severity_results: Dict[int, BenchmarkSummary],
    save_path: Optional[str] = None,
) -> Optional[Any]:
    """
    Line plot: mean accuracy vs. severity level for each method.

    Parameters
    ----------
    severity_results : dict
        {severity_level: BenchmarkSummary} for severities 1-5.
    save_path : str, optional
        Path to save the figure.
    """
    if not _MPL_AVAILABLE:
        return None

    _setup_style()

    severities = sorted(severity_results.keys())
    methods = _METHOD_ORDER

    fig, ax = plt.subplots(figsize=(9, 6))

    for method in methods:
        mean_accs = []
        for sev in severities:
            s = severity_results[sev]
            accs = [
                s.accuracy_table.get(c, {}).get(method, 0.0)
                for c in s.corruption_types
                if c in s.accuracy_table
            ]
            mean_accs.append(np.mean(accs) * 100 if accs else 0)

        ax.plot(
            severities, mean_accs,
            marker="o", linewidth=2.5, markersize=8,
            color=COLOURS.get(method, "#888"),
            label=METHOD_DISPLAY.get(method, method),
            zorder=5,
        )

    ax.set_xlabel("Corruption Severity Level", fontweight="bold")
    ax.set_ylabel("Mean Accuracy (%)", fontweight="bold")
    ax.set_title(
        "Accuracy vs. Corruption Severity — All Methods",
        fontweight="bold", pad=15,
    )
    ax.set_xticks(severities)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.2, zorder=0)
    ax.set_ylim(0, 100)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)
        logger.info("Saved severity analysis -> %s", save_path)
        plt.close(fig)
        return None
    return fig


# ── Convenience: generate all ────────────────────────────────────────────────


def generate_all_figures(
    summary: BenchmarkSummary,
    uncertainty_analyzer=None,
    severity_results: Optional[Dict[int, BenchmarkSummary]] = None,
    output_dir: str = "./figures",
) -> List[str]:
    """
    Generate all available figures and save to output_dir.

    Parameters
    ----------
    summary : BenchmarkSummary
        Results from the benchmark evaluator.
    uncertainty_analyzer : UncertaintyAnalyzer, optional
        For RQ3 scatter plot.
    severity_results : dict, optional
        {severity: BenchmarkSummary} for severity analysis plot.
    output_dir : str
        Directory to save figures.

    Returns
    -------
    list of str
        Paths of all generated figure files.
    """
    if not _MPL_AVAILABLE:
        logger.error("matplotlib required for figure generation.")
        return []

    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    # 1. Accuracy heatmap
    path = os.path.join(output_dir, "accuracy_heatmap.png")
    accuracy_heatmap(summary, save_path=path)
    saved.append(path)

    # 2. mCE bar chart
    path = os.path.join(output_dir, "mce_comparison.png")
    mce_bar_chart(summary, save_path=path)
    saved.append(path)

    # 3. Entropy-gain scatter (RQ3)
    if uncertainty_analyzer is not None:
        path = os.path.join(output_dir, "entropy_gain_scatter.png")
        entropy_gain_scatter(uncertainty_analyzer, save_path=path)
        saved.append(path)

    # 4. Category comparison
    path = os.path.join(output_dir, "category_comparison.png")
    category_comparison(summary, save_path=path)
    saved.append(path)

    # 5. Pseudo-label failure
    path = os.path.join(output_dir, "pl_failure_analysis.png")
    pl_failure_chart(summary, save_path=path)
    saved.append(path)

    # 6. Severity analysis
    if severity_results and len(severity_results) >= 2:
        path = os.path.join(output_dir, "severity_analysis.png")
        severity_analysis(severity_results, save_path=path)
        saved.append(path)

    logger.info("Generated %d figures in %s", len(saved), output_dir)
    return saved
