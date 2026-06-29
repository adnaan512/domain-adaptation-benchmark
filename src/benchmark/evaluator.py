"""
Benchmark Evaluator for the Domain Adaptation Benchmark.

Responsibilities
----------------
1. Aggregate per-(corruption, method) accuracy results.
2. Compute Mean Corruption Error (mCE) for each method.
3. Compute relative mCE improvement vs. the no-adaptation baseline.
4. Identify the winner (best method) per corruption type.
5. Detect pseudo-label confirmation bias failure on blur corruptions.
6. Generate ASCII heatmap, mCE summary, and winner tables.

mCE Definition
--------------
Hendrycks & Dietterich (2019) define Corruption Error (CE) relative to an
AlexNet baseline.  In this benchmark (without the AlexNet normalisation) we
use the simplified version:

    mCE = (1 / |C|) Σ_{c ∈ C} (1 − accuracy_c)

where C is the set of evaluated corruption types.  Lower mCE is better.

Relative improvement over baseline:
    RelImp = (mCE_baseline − mCE_method) / mCE_baseline

A positive RelImp means the method reduces corruption error compared to
the no-adaptation baseline.

ASCII Heatmap Symbols
---------------------
    ▓  accuracy > 80%     (strong)
    ▒  accuracy 60–80%    (moderate)
    ░  accuracy < 60%     (weak)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from src.models import (
    ALL_CORRUPTIONS,
    BLUR_CORRUPTIONS,
    METHOD_DISPLAY,
    METHODS,
    BenchmarkSummary,
    CorruptionProfile,
    get_category,
)

logger = logging.getLogger(__name__)

# ASCII density blocks for the heatmap
_BLOCK_HIGH = "▓"   # > 80%
_BLOCK_MED  = "▒"   # 60–80%
_BLOCK_LOW  = "░"   # < 60%

_METHOD_ORDER = ["no_adaptation", "test_time_norm", "tent", "pseudo_label"]


def _acc_to_block(acc: Optional[float]) -> str:
    if acc is None:
        return "  N/A  "
    if acc > 0.80:
        symbol = _BLOCK_HIGH
    elif acc > 0.60:
        symbol = _BLOCK_MED
    else:
        symbol = _BLOCK_LOW
    return f"{symbol}{acc:6.2%}{symbol}"


class BenchmarkEvaluator:
    """
    Accumulates per-(corruption, method) results and computes benchmark metrics.

    Parameters
    ----------
    corruption_types : list[str]
        Ordered list of corruption types to evaluate.
        Each type must be one of ALL_CORRUPTIONS (or a mock subset for demo).
    severity : int
        Severity level this evaluator instance covers.
    """

    def __init__(
        self,
        corruption_types: List[str],
        severity: int = 3,
    ) -> None:
        self.corruption_types = list(corruption_types)
        self.severity         = severity

        # Internal storage
        self._accuracy_table: Dict[str, Dict[str, float]] = {}
        self._entropy_table:  Dict[str, Dict[str, float]] = {}

    # ------------------------------------------------------------------ #
    # Data ingestion                                                       #
    # ------------------------------------------------------------------ #

    def add_result(
        self,
        corruption_type: str,
        method: str,
        accuracy: float,
        entropy_before: float = 0.0,
    ) -> None:
        """
        Record a (corruption, method) accuracy result.

        Parameters
        ----------
        corruption_type : str
            Name of the corruption (e.g., 'gaussian_noise').
        method : str
            One of: no_adaptation, test_time_norm, tent, pseudo_label.
        accuracy : float
            Top-1 accuracy in [0, 1].
        entropy_before : float
            Mean Shannon entropy before adaptation for this (corruption, method).
        """
        if corruption_type not in self._accuracy_table:
            self._accuracy_table[corruption_type] = {}
            self._entropy_table[corruption_type]  = {}

        self._accuracy_table[corruption_type][method] = accuracy
        self._entropy_table[corruption_type][method]  = entropy_before

        logger.debug(
            "Recorded: %s / %s → acc=%.4f  H=%.4f",
            corruption_type, method, accuracy, entropy_before,
        )

    # ------------------------------------------------------------------ #
    # Metric computation                                                   #
    # ------------------------------------------------------------------ #

    def _compute_winners(self) -> Dict[str, str]:
        """Return {corruption: best_method} mapping."""
        winners: Dict[str, str] = {}
        for corruption, method_accs in self._accuracy_table.items():
            if method_accs:
                winners[corruption] = max(method_accs, key=method_accs.get)  # type: ignore[arg-type]
        return winners

    def _compute_mce(self) -> Dict[str, float]:
        """Compute mCE for each method (mean error rate across corruptions)."""
        mce: Dict[str, float] = {}
        for method in _METHOD_ORDER:
            errors = [
                1.0 - self._accuracy_table[c][method]
                for c in self.corruption_types
                if c in self._accuracy_table
                and method in self._accuracy_table[c]
            ]
            if errors:
                mce[method] = sum(errors) / len(errors)
        return mce

    def _compute_relative_improvements(
        self, mce: Dict[str, float]
    ) -> Dict[str, float]:
        """Relative improvement over no-adaptation baseline."""
        baseline = mce.get("no_adaptation", None)
        if baseline is None or baseline == 0.0:
            return {m: 0.0 for m in mce}
        return {
            m: (baseline - mce_val) / baseline
            for m, mce_val in mce.items()
        }

    def _build_profiles(self, winners: Dict[str, str]) -> Dict[str, CorruptionProfile]:
        """Build CorruptionProfile for each evaluated corruption."""
        profiles: Dict[str, CorruptionProfile] = {}
        for corruption in self.corruption_types:
            accs    = self._accuracy_table.get(corruption, {})
            entropies = self._entropy_table.get(corruption, {})
            baseline_acc = accs.get("no_adaptation", 0.0)
            baseline_h   = entropies.get("no_adaptation", 0.0)
            profile = CorruptionProfile(
                corruption_type   = corruption,
                severity          = self.severity,
                category          = get_category(corruption),
                mean_entropy      = baseline_h,
                mean_confidence   = 0.0,
                accuracy_no_adapt = baseline_acc,
                accuracies        = dict(accs),
                entropies         = dict(entropies),
                winner            = winners.get(corruption, ""),
            )
            profiles[corruption] = profile
        return profiles

    def _detect_pl_blur_failures(self) -> Dict[str, Dict[str, float]]:
        """
        Identify blur corruptions where pseudo-label accuracy < baseline.

        Returns
        -------
        dict
            {corruption: {no_adaptation, pseudo_label, degradation}}
            for each blur corruption where PL underperforms baseline.
        """
        failures: Dict[str, Dict[str, float]] = {}
        for corruption in BLUR_CORRUPTIONS:
            if corruption not in self._accuracy_table:
                continue
            baseline = self._accuracy_table[corruption].get("no_adaptation")
            pl_acc   = self._accuracy_table[corruption].get("pseudo_label")
            if baseline is not None and pl_acc is not None and pl_acc < baseline:
                failures[corruption] = {
                    "no_adaptation": baseline,
                    "pseudo_label":  pl_acc,
                    "degradation":   baseline - pl_acc,
                }
        return failures

    # ------------------------------------------------------------------ #
    # Finalisation                                                         #
    # ------------------------------------------------------------------ #

    def finalize(self) -> BenchmarkSummary:
        """
        Compute all aggregate metrics and return a BenchmarkSummary.

        Returns
        -------
        BenchmarkSummary
            Complete results container used by ReportGenerator.
        """
        winners          = self._compute_winners()
        mce              = self._compute_mce()
        rel_improvements = self._compute_relative_improvements(mce)
        profiles         = self._build_profiles(winners)
        pl_failures      = self._detect_pl_blur_failures()

        # Log mCE table
        logger.info("mCE results:")
        for method, val in mce.items():
            imp = rel_improvements.get(method, 0.0)
            logger.info(
                "  %-20s mCE=%.4f  RelImp=%+.2f%%",
                METHOD_DISPLAY.get(method, method), val, 100 * imp,
            )
        if pl_failures:
            logger.warning(
                "Pseudo-label blur failures detected on: %s",
                list(pl_failures.keys()),
            )

        summary = BenchmarkSummary(
            methods              = _METHOD_ORDER,
            corruption_types     = self.corruption_types,
            severity             = self.severity,
            mce_scores           = mce,
            relative_improvements = rel_improvements,
            accuracy_table       = {c: dict(m) for c, m in self._accuracy_table.items()},
            entropy_table        = {c: dict(m) for c, m in self._entropy_table.items()},
            winners              = winners,
            profiles             = profiles,
            pseudo_label_blur_failures = pl_failures,
        )
        # Store reference for text-report helpers
        self.results = summary
        return summary

    # ------------------------------------------------------------------ #
    # Text report helpers                                                  #
    # ------------------------------------------------------------------ #

    def format_ascii_heatmap(self) -> str:
        """
        Generate a 15×4 (corruption × method) ASCII heatmap.

        Symbols: ▓ >80%  ▒ 60-80%  ░ <60%
        """
        col_labels = [METHOD_DISPLAY.get(m, m) for m in _METHOD_ORDER]
        header     = (
            f"{'Corruption':<26}"
            + "".join(f"{c:^10}" for c in col_labels)
            + "  Winner"
        )
        sep = "─" * (26 + 10 * 4 + 10)

        lines = [
            "CORRUPTION × METHOD ACCURACY HEATMAP",
            f"  ▓ >80%  ▒ 60-80%  ░ <60%",
            "",
            header,
            sep,
        ]

        for corruption in self.corruption_types:
            accs   = self._accuracy_table.get(corruption, {})
            winner = self._compute_winners().get(corruption, "N/A")

            row = f"{corruption:<26}"
            for method in _METHOD_ORDER:
                acc  = accs.get(method)
                cell = _acc_to_block(acc)
                row += f"{cell:^10}"
            row += f"  {METHOD_DISPLAY.get(winner, winner)}"
            lines.append(row)

        lines.append(sep)

        # mCE footer
        mce = self._compute_mce()
        mce_row = f"{'mCE (↓ better)':<26}"
        for method in _METHOD_ORDER:
            val = mce.get(method)
            mce_row += f"{'N/A' if val is None else f'{val:.4f}':^10}"
        lines.append(mce_row)

        rel = self._compute_relative_improvements(mce)
        imp_row = f"{'Rel. Improve':<26}"
        for method in _METHOD_ORDER:
            val = rel.get(method, 0.0)
            imp_row += f"{f'{val:+.1%}':^10}"
        lines.append(imp_row)

        return "\n".join(lines)

    def format_mce_table(self) -> str:
        """Return a clean plain-text mCE summary table."""
        mce = self._compute_mce()
        rel = self._compute_relative_improvements(mce)
        lines = [
            "Mean Corruption Error (mCE) Summary",
            "═" * 52,
            f"{'Method':<22} {'mCE':>8}   {'Relative Improvement':>20}",
            "─" * 52,
        ]
        for method in _METHOD_ORDER:
            if method not in mce:
                continue
            lines.append(
                f"{METHOD_DISPLAY.get(method, method):<22} "
                f"{mce[method]:>8.4f}   "
                f"{rel.get(method, 0.0):>+19.2%}"
            )
        return "\n".join(lines)

    def format_winner_table(self) -> str:
        """Return a per-corruption winner table."""
        winners = self._compute_winners()
        lines   = [
            "Winner per Corruption Type",
            "═" * 58,
            f"{'Corruption':<26} {'Winner':<16} {'Accuracy':>9}",
            "─" * 58,
        ]
        for corruption in self.corruption_types:
            winner = winners.get(corruption, "N/A")
            acc    = self._accuracy_table.get(corruption, {}).get(winner, 0.0)
            lines.append(
                f"{corruption:<26} "
                f"{METHOD_DISPLAY.get(winner, winner):<16} "
                f"{acc:>9.2%}"
            )
        return "\n".join(lines)

    def generate_full_report(self) -> str:
        """Generate the complete text benchmark report."""
        pl_failures = self._detect_pl_blur_failures()

        sections = [
            "═" * 72,
            "DOMAIN ADAPTATION BENCHMARK — RESULTS",
            "═" * 72,
            "",
            self.format_ascii_heatmap(),
            "",
            self.format_mce_table(),
            "",
            self.format_winner_table(),
        ]

        if pl_failures:
            sections += [
                "",
                "⚠  FINDING: Pseudo-Label Confirmation Bias on Blur Corruptions",
                "─" * 65,
                f"{'Corruption':<26} {'Baseline':>10} {'PL Acc':>10} {'Δ':>10}",
                "─" * 65,
            ]
            for c, data in pl_failures.items():
                sections.append(
                    f"{c:<26} "
                    f"{data['no_adaptation']:>10.2%} "
                    f"{data['pseudo_label']:>10.2%} "
                    f"{-data['degradation']:>+9.2%}"
                )
            sections += [
                "",
                "Cause: Blur corruptions produce confident but WRONG predictions.",
                "These pass the 0.9 threshold and serve as incorrect pseudo-labels,",
                "causing the model to fine-tune toward wrong classes (confirmation bias).",
            ]

        return "\n".join(sections)
