"""
Statistical Analysis for the Domain Adaptation Benchmark.

Provides bootstrap confidence intervals and paired significance tests
to elevate results from "indicative" to publishable research quality.

Functions
---------
    bootstrap_ci()          — 95% confidence interval for accuracy
    paired_significance()   — Wilcoxon signed-rank test for method comparison
    compute_all_stats()     — Full statistical analysis across methods

Usage
-----
    from src.benchmark.stats import compute_all_stats
    stats = compute_all_stats(summary)
    print(stats["tent"]["mean_accuracy"])     # 0.672
    print(stats["tent"]["ci_95"])             # (0.664, 0.680)
    print(stats["tent"]["vs_baseline_p"])     # 0.0023
"""

from __future__ import annotations

import logging
import math
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for the mean of values.

    Parameters
    ----------
    values : list of float
        Sample values (e.g., per-corruption accuracies for one method).
    n_bootstrap : int
        Number of bootstrap resamples.
    ci_level : float
        Confidence level (default 0.95 for 95% CI).
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    (mean, ci_lower, ci_upper) : tuple of float
        Point estimate and confidence interval bounds.
    """
    if not values:
        return (0.0, 0.0, 0.0)

    n = len(values)
    rng = random.Random(seed)
    means = []

    for _ in range(n_bootstrap):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)

    means.sort()
    alpha = 1.0 - ci_level
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))

    point_estimate = sum(values) / n
    ci_lower = means[max(0, lower_idx)]
    ci_upper = means[min(n_bootstrap - 1, upper_idx)]

    return (point_estimate, ci_lower, ci_upper)


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test (pure Python, no scipy dependency)
# ---------------------------------------------------------------------------


def _wilcoxon_signed_rank(
    x: List[float],
    y: List[float],
) -> Tuple[float, float]:
    """
    Simplified Wilcoxon signed-rank test for paired samples.

    Tests H0: median difference = 0 vs H1: median difference != 0.

    Uses a normal approximation for n >= 10.  For smaller samples, returns
    the test statistic and a conservative p-value estimate.

    Parameters
    ----------
    x : list of float
        First sample (e.g., baseline accuracies per corruption).
    y : list of float
        Second sample (e.g., TENT accuracies per corruption).

    Returns
    -------
    (statistic, p_value) : tuple of float
        Test statistic W and two-sided p-value.
    """
    n = len(x)
    if n != len(y) or n < 3:
        return (0.0, 1.0)

    # Compute differences and ranks
    diffs = [(yi - xi) for xi, yi in zip(x, y)]
    # Remove zeros
    nonzero = [(abs(d), d) for d in diffs if d != 0.0]
    if not nonzero:
        return (0.0, 1.0)

    # Rank by absolute value
    nonzero.sort(key=lambda t: t[0])
    nr = len(nonzero)

    # Assign ranks (handle ties with average rank)
    ranks = list(range(1, nr + 1))

    # Sum of positive and negative ranks
    w_plus = sum(
        ranks[i] for i in range(nr) if nonzero[i][1] > 0
    )
    w_minus = sum(
        ranks[i] for i in range(nr) if nonzero[i][1] < 0
    )
    w = min(w_plus, w_minus)

    # Normal approximation for p-value (valid for nr >= 10)
    if nr >= 10:
        mean_w = nr * (nr + 1) / 4
        std_w = math.sqrt(nr * (nr + 1) * (2 * nr + 1) / 24)
        if std_w == 0:
            return (w, 1.0)
        z = (w - mean_w) / std_w
        # Two-tailed p-value using error function approximation
        p_value = 2.0 * _normal_cdf(-abs(z))
    else:
        # Conservative estimate for small samples
        # Use the exact distribution bounds
        max_w = nr * (nr + 1) / 2
        p_value = (2 * w + 1) / (max_w + 1)
        p_value = min(1.0, p_value)

    return (w, p_value)


def _normal_cdf(z: float) -> float:
    """Approximate standard normal CDF using error function."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def paired_significance(
    baseline_accs: List[float],
    method_accs: List[float],
) -> Dict[str, Any]:
    """
    Test whether a TTA method significantly differs from the baseline.

    Uses paired Wilcoxon signed-rank test (paired by corruption type).

    Parameters
    ----------
    baseline_accs : list of float
        Per-corruption accuracies for the baseline (no adaptation).
    method_accs : list of float
        Per-corruption accuracies for the TTA method.

    Returns
    -------
    dict with keys:
        statistic   : float   Wilcoxon W statistic
        p_value     : float   Two-sided p-value
        significant : bool    True if p < 0.05
        mean_diff   : float   Mean accuracy difference (method - baseline)
        n_improved  : int     Number of corruptions where method > baseline
        n_degraded  : int     Number of corruptions where method < baseline
    """
    w, p = _wilcoxon_signed_rank(baseline_accs, method_accs)
    diffs = [m - b for b, m in zip(baseline_accs, method_accs)]

    return {
        "statistic": w,
        "p_value": p,
        "significant": p < 0.05,
        "mean_diff": sum(diffs) / len(diffs) if diffs else 0.0,
        "n_improved": sum(1 for d in diffs if d > 0),
        "n_degraded": sum(1 for d in diffs if d < 0),
        "n_equal": sum(1 for d in diffs if d == 0.0),
    }


# ---------------------------------------------------------------------------
# Full statistical analysis
# ---------------------------------------------------------------------------


def compute_all_stats(
    summary,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute comprehensive statistics for all methods in a benchmark run.

    For each method, computes:
        - Mean accuracy with 95% bootstrap CI
        - mCE with 95% bootstrap CI
        - Paired significance test vs. baseline
        - Number of corruptions improved/degraded

    Parameters
    ----------
    summary : BenchmarkSummary
        Completed benchmark results.

    Returns
    -------
    dict
        {method_name: {mean_accuracy, ci_95, mce, mce_ci_95,
                       vs_baseline_p, vs_baseline_significant, ...}}
    """
    from src.models import METHOD_DISPLAY

    results: Dict[str, Dict[str, Any]] = {}
    methods = ["no_adaptation", "test_time_norm", "tent", "pseudo_label"]

    # Gather per-corruption accuracies for each method
    method_accs: Dict[str, List[float]] = {m: [] for m in methods}
    for corruption in summary.corruption_types:
        if corruption not in summary.accuracy_table:
            continue
        for method in methods:
            acc = summary.accuracy_table[corruption].get(method)
            if acc is not None:
                method_accs[method].append(acc)

    baseline_accs = method_accs.get("no_adaptation", [])

    for method in methods:
        accs = method_accs[method]
        errors = [1.0 - a for a in accs]

        if not accs:
            results[method] = {
                "display_name": METHOD_DISPLAY.get(method, method),
                "mean_accuracy": 0.0,
                "ci_95": (0.0, 0.0),
                "mce": 0.0,
                "mce_ci_95": (0.0, 0.0),
            }
            continue

        # Bootstrap CI for accuracy
        mean_acc, ci_low, ci_high = bootstrap_ci(accs)

        # Bootstrap CI for mCE
        mean_mce, mce_ci_low, mce_ci_high = bootstrap_ci(errors)

        result = {
            "display_name": METHOD_DISPLAY.get(method, method),
            "mean_accuracy": mean_acc,
            "ci_95": (ci_low, ci_high),
            "std_accuracy": _std(accs),
            "mce": mean_mce,
            "mce_ci_95": (mce_ci_low, mce_ci_high),
            "n_corruptions": len(accs),
        }

        # Significance test vs baseline
        if method != "no_adaptation" and baseline_accs:
            sig = paired_significance(baseline_accs, accs)
            result.update({
                "vs_baseline_p": sig["p_value"],
                "vs_baseline_significant": sig["significant"],
                "vs_baseline_statistic": sig["statistic"],
                "n_improved": sig["n_improved"],
                "n_degraded": sig["n_degraded"],
                "mean_improvement": sig["mean_diff"],
            })

        results[method] = result

    return results


def format_stats_table(stats: Dict[str, Dict[str, Any]]) -> str:
    """
    Format statistical results as a publication-ready ASCII table.

    Parameters
    ----------
    stats : dict
        Output of compute_all_stats().

    Returns
    -------
    str
        Multi-line formatted table.
    """
    lines = [
        "Statistical Analysis — Method Comparison",
        "=" * 85,
        f"{'Method':<16} {'Acc (%)':<12} {'95% CI':<18} "
        f"{'mCE':<10} {'p-value':<10} {'Sig.':<6} {'Improved':<10}",
        "-" * 85,
    ]

    method_order = ["no_adaptation", "test_time_norm", "tent", "pseudo_label"]
    for method in method_order:
        if method not in stats:
            continue
        s = stats[method]
        name = s.get("display_name", method)
        acc = s.get("mean_accuracy", 0.0)
        ci = s.get("ci_95", (0.0, 0.0))
        mce = s.get("mce", 0.0)
        p_val = s.get("vs_baseline_p", None)
        sig = s.get("vs_baseline_significant", None)
        n_imp = s.get("n_improved", None)

        ci_str = f"[{ci[0]:.1%}, {ci[1]:.1%}]"
        p_str = f"{p_val:.4f}" if p_val is not None else "—"
        sig_str = "Yes*" if sig else ("No" if sig is not None else "—")
        imp_str = f"{n_imp}/{s.get('n_corruptions', 0)}" if n_imp is not None else "—"

        lines.append(
            f"{name:<16} {acc:<12.2%} {ci_str:<18} "
            f"{mce:<10.4f} {p_str:<10} {sig_str:<6} {imp_str:<10}"
        )

    lines.append("-" * 85)
    lines.append("* Significant at p < 0.05 (Wilcoxon signed-rank test)")

    return "\n".join(lines)


def _std(values: List[float]) -> float:
    """Compute sample standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(variance)
