"""
Uncertainty Analysis: Pre-Adaptation Entropy as a Predictor of TTA Benefit.

Hypothesis (RQ3)
----------------
If Shannon entropy H(p) measures how "surprised" a model is by its input,
then high pre-adaptation entropy signals a large distribution shift between
the source (clean) and target (corrupted) domains.

Since TTA methods adapt to the test distribution, they should provide
larger accuracy improvements where the shift is larger — i.e., where
pre-adaptation entropy is highest.

Formally, we test:
    Is Pearson r(H_pre, ΔAcc) > 0 ?

where:
    H_pre  = mean entropy of model predictions on a corruption BEFORE adaptation
    ΔAcc   = acc_after_TENT - acc_before_TENT  (TENT adaptation gain)
    r      = Pearson correlation coefficient across the 15 corruption types

Expected finding from the literature:
    Noise corruptions (gaussian_noise, shot_noise, impulse_noise) cause the
    model to produce high-entropy predictions (the model is genuinely uncertain
    about noisy images).  Digital corruptions such as brightness and contrast
    often preserve class-discriminative features, so entropy stays lower.
    
    Predicted ranking: noise > blur > weather > digital  (entropy order)
    
    If the hypothesis holds, TENT should improve noise corruptions more than
    digital corruptions, and r should be significantly positive (> 0.4).

Methodology
-----------
    1. For each corruption type at severity 3 (moderate, representative):
       a. Run the unmodified model on 10 000 test images.
       b. Compute mean Shannon entropy H̄_pre over all predictions.
    2. Run TENT on the same loader; compute ΔAcc = TENT_acc - baseline_acc.
    3. Compute Pearson r between H̄_pre and ΔAcc across corruption types.
    4. Rank corruptions by H̄_pre; surface the ranking in the report.

Why severity 3?
    Severity 3 (middle of the 1–5 range) represents a moderate shift — not
    so mild that all methods succeed trivially, not so severe that all methods
    fail uniformly.  Results at severity 3 best discriminate method quality.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure-Python Pearson correlation (no numpy dependency)
# ---------------------------------------------------------------------------


def pearson_correlation(x: List[float], y: List[float]) -> float:
    """
    Compute Pearson product-moment correlation coefficient.

    r = Σ((xᵢ - x̄)(yᵢ - ȳ)) / sqrt(Σ(xᵢ - x̄)² · Σ(yᵢ - ȳ)²)

    Returns 0.0 if either variable has zero variance or fewer than 2 points.

    Parameters
    ----------
    x : list of float
        First variable (e.g., pre-adaptation entropies).
    y : list of float
        Second variable (e.g., adaptation gains).

    Returns
    -------
    float
        Pearson r in [-1, 1].
    """
    n = len(x)
    if n != len(y) or n < 2:
        return 0.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    cov    = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    var_x  = sum((xi - mean_x) ** 2 for xi in x)
    var_y  = sum((yi - mean_y) ** 2 for yi in y)

    if var_x == 0.0 or var_y == 0.0:
        return 0.0

    return cov / math.sqrt(var_x * var_y)


# ---------------------------------------------------------------------------
# Core entropy computation
# ---------------------------------------------------------------------------


def compute_pre_adaptation_entropy(
    model: nn.Module,
    data_loader,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Compute pre-adaptation uncertainty metrics for a given DataLoader.

    The model is set to eval mode and no gradients are computed.

    Parameters
    ----------
    model : nn.Module
        Pre-trained model (before any adaptation for this corruption).
    data_loader : DataLoader
        DataLoader for one corruption type at one severity.
    device : torch.device
        Computation device.

    Returns
    -------
    dict with keys:
        mean_entropy    : float   mean H(p) in nats
        std_entropy     : float   standard deviation of H(p) across samples
        mean_max_prob   : float   mean max-softmax probability
        fraction_uncertain : float  fraction of samples with max_prob < 0.5
        num_samples     : int
    """
    model.eval()

    all_entropies: List[float] = []
    all_max_probs: List[float] = []

    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs  = torch.softmax(logits, dim=1)

            log_p     = torch.log(probs + 1e-8)
            entropy   = -(probs * log_p).sum(dim=1)
            max_probs = probs.max(dim=1).values

            all_entropies.extend(entropy.cpu().tolist())
            all_max_probs.extend(max_probs.cpu().tolist())

    e_t = torch.tensor(all_entropies)
    m_t = torch.tensor(all_max_probs)

    return {
        "mean_entropy":      float(e_t.mean().item()),
        "std_entropy":       float(e_t.std().item()),
        "mean_max_prob":     float(m_t.mean().item()),
        "fraction_uncertain": float((m_t < 0.5).float().mean().item()),
        "num_samples":       len(all_entropies),
    }


# ---------------------------------------------------------------------------
# Analyzer class
# ---------------------------------------------------------------------------


class UncertaintyAnalyzer:
    """
    Tracks pre-adaptation entropy and TTA gain across corruption types,
    then computes the Pearson correlation to test the hypothesis:
        high entropy → larger TENT benefit.

    Usage
    -----
    ::

        analyzer = UncertaintyAnalyzer(model, device)

        for corruption in corruption_types:
            loader = get_loader(corruption, severity=3)
            metrics = analyzer.compute_entropy(loader)
            analyzer.record(corruption, metrics, baseline_acc, tent_acc)

        r = analyzer.compute_correlation()
        print(analyzer.generate_report())

    Parameters
    ----------
    model : nn.Module
        Pre-trained model (before adaptation).  Must be restored to original
        state before each call to ``compute_entropy()``.
    device : torch.device
        Computation device.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model   = model
        self.device  = device
        self._records: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def compute_entropy(self, data_loader) -> Dict[str, Any]:
        """
        Compute pre-adaptation entropy for a single DataLoader.

        The model must already be in its unmodified state (call
        ``model.restore_original_state()`` + ``model.eval()`` before this).
        """
        return compute_pre_adaptation_entropy(self.model, data_loader, self.device)

    def record(
        self,
        corruption_type: str,
        entropy_metrics: Dict[str, Any],
        baseline_accuracy: float,
        tent_accuracy: float,
    ) -> None:
        """
        Record metrics for one corruption type.

        Parameters
        ----------
        corruption_type : str
            Name of the corruption (e.g., 'gaussian_noise').
        entropy_metrics : dict
            Output of ``compute_entropy()``.
        baseline_accuracy : float
            Top-1 accuracy without adaptation.
        tent_accuracy : float
            Top-1 accuracy after TENT adaptation.
        """
        adaptation_gain = tent_accuracy - baseline_accuracy
        self._records[corruption_type] = {
            **entropy_metrics,
            "baseline_accuracy": baseline_accuracy,
            "tent_accuracy":     tent_accuracy,
            "adaptation_gain":   adaptation_gain,
        }

    def compute_correlation(self) -> float:
        """
        Compute Pearson r between mean pre-adaptation entropy and TENT gain.

        Returns
        -------
        float
            Pearson r in [-1, 1].  Returns 0.0 if fewer than 2 corruptions recorded.
        """
        if len(self._records) < 2:
            return 0.0

        entropies = [r["mean_entropy"]   for r in self._records.values()]
        gains     = [r["adaptation_gain"] for r in self._records.values()]

        r = pearson_correlation(entropies, gains)
        logger.info("Entropy–Adaptation Gain Pearson r = %.4f", r)
        return r

    def get_ranked_corruptions(self) -> List[Tuple[str, float, float]]:
        """
        Return corruption types sorted by pre-adaptation entropy (descending).

        Returns
        -------
        list of (corruption_type, mean_entropy, adaptation_gain)
        """
        return sorted(
            [
                (k, v["mean_entropy"], v["adaptation_gain"])
                for k, v in self._records.items()
            ],
            key=lambda t: t[1],
            reverse=True,
        )

    def get_records(self) -> Dict[str, Dict[str, Any]]:
        """Return the full internal records dictionary (read-only copy)."""
        return dict(self._records)

    def generate_report(self) -> str:
        """
        Generate a human-readable text report of entropy vs. adaptation gain.

        Returns
        -------
        str
            Multi-line formatted report.
        """
        r      = self.compute_correlation()
        ranked = self.get_ranked_corruptions()

        lines: List[str] = [
            "=" * 70,
            "UNCERTAINTY ANALYSIS — Pre-Adaptation Entropy vs. TENT Gain (RQ3)",
            "=" * 70,
            f"Pearson r (entropy ↔ TENT adaptation gain):  {r:+.4f}",
            "",
            f"{'Corruption':<25} {'H̄_pre':>8} {'ΔAcc':>8} {'Base':>8} {'TENT':>8} {'Uncertain%':>11}",
            "-" * 70,
        ]

        for corruption, entropy, gain in ranked:
            rec = self._records[corruption]
            frac_unc = rec.get("fraction_uncertain", 0.0)
            lines.append(
                f"{corruption:<25} "
                f"{entropy:>8.4f} "
                f"{gain:>+8.4f} "
                f"{rec['baseline_accuracy']:>7.2%} "
                f"{rec['tent_accuracy']:>7.2%} "
                f"{frac_unc:>10.1%}"
            )

        lines += [
            "",
            "Interpretation:",
        ]

        if r > 0.5:
            lines += [
                f"  r = {r:+.4f} → Strong positive correlation.",
                "  Corruptions with higher pre-adaptation entropy benefit more from TENT.",
                "  This supports RQ3: entropy predicts which corruptions benefit from TTA.",
                "  Expected: noise corruptions (high entropy) > digital (low entropy).",
            ]
        elif r > 0.2:
            lines += [
                f"  r = {r:+.4f} → Moderate positive correlation.",
                "  Entropy partially predicts TENT benefit across corruption types.",
            ]
        elif r < -0.2:
            lines += [
                f"  r = {r:+.4f} → Negative correlation (unexpected).",
                "  High-entropy corruptions do NOT benefit more from TENT.",
                "  Possible cause: model collapse on very high-entropy inputs.",
            ]
        else:
            lines += [
                f"  r = {r:+.4f} → Weak / no correlation.",
                "  Entropy is not a reliable predictor of TTA benefit for this architecture.",
                "  Consider: batch size may be too small for reliable entropy estimates.",
            ]

        return "\n".join(lines)

    def generate_ascii_scatter(self, width: int = 60, height: int = 15) -> str:
        """
        Generate a simple ASCII scatter plot of entropy vs. adaptation gain.

        Each point (·) represents one corruption type.
        Axis labels show min/max of each variable.

        Parameters
        ----------
        width : int
            Plot width in characters.
        height : int
            Plot height in lines.

        Returns
        -------
        str
            Multi-line ASCII scatter plot.
        """
        if not self._records:
            return "(no data)"

        ranked  = self.get_ranked_corruptions()
        xs      = [t[1] for t in ranked]   # entropy
        ys      = [t[2] for t in ranked]   # adaptation gain
        labels  = [t[0][:3] for t in ranked]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        x_range      = x_max - x_min or 1.0
        y_range      = y_max - y_min or 1.0

        # Map to grid coordinates
        grid = [[" "] * width for _ in range(height)]

        for label, xi, yi in zip(labels, xs, ys):
            col = int((xi - x_min) / x_range * (width  - 1))
            row = int((1 - (yi - y_min) / y_range) * (height - 1))
            col = max(0, min(width  - 1, col))
            row = max(0, min(height - 1, row))
            if grid[row][col] == " ":
                grid[row][col] = "·"
            else:
                grid[row][col] = "+"   # overlap marker

        lines: List[str] = [
            f"  ΔAcc↑  (y_max={y_max:+.3f})",
        ]
        for i, row in enumerate(grid):
            prefix = f"  {y_max - (y_max - y_min) * i / (height - 1):+.3f} │" if i == 0 else "         │"
            lines.append(prefix + "".join(row))
        lines.append("         └" + "─" * width)
        lines.append(f"          {x_min:.3f}{' ' * (width - 14)}{x_max:.3f}")
        lines.append(f"          {'H̄_pre →' : <{width}}")

        return "\n".join(lines)
