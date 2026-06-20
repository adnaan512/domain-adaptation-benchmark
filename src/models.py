"""
Data models for the Domain Adaptation Benchmark.

These dataclasses are the canonical data contracts between pipeline stages:
    - AdaptationResult:   output of a single (corruption, method) run
    - CorruptionProfile:  aggregated stats for one corruption type
    - UncertaintyMetrics: pre-adaptation entropy metrics per corruption
    - BenchmarkSummary:   top-level results container for reporting
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Method registry
# ---------------------------------------------------------------------------

METHODS = ["no_adaptation", "test_time_norm", "tent", "pseudo_label"]

METHOD_DISPLAY: Dict[str, str] = {
    "no_adaptation": "No Adapt",
    "test_time_norm": "TTN",
    "tent": "TENT",
    "pseudo_label": "Pseudo-Label",
}

CORRUPTION_CATEGORIES: Dict[str, List[str]] = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

ALL_CORRUPTIONS: List[str] = [c for cats in CORRUPTION_CATEGORIES.values() for c in cats]

BLUR_CORRUPTIONS = set(CORRUPTION_CATEGORIES["blur"])


def get_category(corruption_type: str) -> str:
    """Return the corruption category (noise/blur/weather/digital)."""
    for cat, members in CORRUPTION_CATEGORIES.items():
        if corruption_type in members:
            return cat
    return "unknown"


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AdaptationResult:
    """
    Output of running one TTA method on one (corruption, severity) pair.

    Fields
    ------
    corruption_type : str
        One of the 15 CIFAR-10-C corruption types.
    severity : int
        Integer 1–5 (1 = mild, 5 = severe).
    method : str
        One of: 'no_adaptation', 'test_time_norm', 'tent', 'pseudo_label'.
    accuracy : float
        Top-1 accuracy in [0, 1].
    loss : float
        Mean cross-entropy loss over the evaluation set.
    entropy_before : float
        Mean Shannon entropy (nats) of model predictions before adaptation.
    entropy_after : float
        Mean Shannon entropy (nats) after adaptation. Equal to entropy_before
        for no_adaptation.
    num_samples : int
        Total number of test samples evaluated.
    adaptation_gain : float
        accuracy - baseline_accuracy for this corruption.  Set externally
        by the evaluator after the baseline result is known.
    extra_metrics : dict
        Method-specific diagnostics (e.g., gradient norms for TENT,
        acceptance rate for pseudo-label).
    """

    corruption_type: str
    severity: int
    method: str
    accuracy: float
    loss: float
    entropy_before: float
    entropy_after: float
    num_samples: int
    adaptation_gain: float = 0.0
    extra_metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def error_rate(self) -> float:
        """1 - accuracy (used for mCE computation)."""
        return 1.0 - self.accuracy

    @property
    def entropy_reduction(self) -> float:
        """Positive value means entropy decreased after adaptation."""
        return self.entropy_before - self.entropy_after

    def __repr__(self) -> str:
        return (
            f"AdaptationResult("
            f"{self.corruption_type}/{self.severity}, "
            f"method={self.method}, "
            f"acc={self.accuracy:.4f}, "
            f"Δentropy={self.entropy_reduction:+.4f})"
        )


@dataclass
class CorruptionProfile:
    """
    Aggregated statistics for a single corruption type across all methods.

    Built by the evaluator from individual AdaptationResult objects.

    Fields
    ------
    corruption_type : str
        Name of the corruption.
    severity : int
        Severity level evaluated.
    category : str
        Category: noise | blur | weather | digital.
    mean_entropy : float
        Pre-adaptation entropy (from no-adaptation baseline).
    mean_confidence : float
        Mean max-softmax probability before adaptation.
    accuracy_no_adapt : float
        Baseline accuracy with no adaptation.
    accuracies : dict
        {method: accuracy} for all evaluated methods.
    entropies : dict
        {method: mean_entropy_before} for all methods.
    winner : str
        Method achieving highest accuracy on this corruption.
    """

    corruption_type: str
    severity: int
    category: str = ""
    mean_entropy: float = 0.0
    mean_confidence: float = 0.0
    accuracy_no_adapt: float = 0.0
    accuracies: Dict[str, float] = field(default_factory=dict)
    entropies: Dict[str, float] = field(default_factory=dict)
    winner: str = ""

    def __post_init__(self) -> None:
        if not self.category:
            self.category = get_category(self.corruption_type)

    def compute_winner(self) -> str:
        """Determine and store the best method for this corruption."""
        if self.accuracies:
            self.winner = max(self.accuracies, key=self.accuracies.get)  # type: ignore[arg-type]
        return self.winner

    @property
    def best_accuracy(self) -> float:
        return max(self.accuracies.values()) if self.accuracies else 0.0

    @property
    def worst_accuracy(self) -> float:
        return min(self.accuracies.values()) if self.accuracies else 0.0

    @property
    def improvement_from_best(self) -> float:
        """Accuracy lift from best TTA vs no-adaptation baseline."""
        return self.best_accuracy - self.accuracy_no_adapt


@dataclass
class UncertaintyMetrics:
    """
    Pre-adaptation entropy metrics for one corruption type.

    Used by UncertaintyAnalyzer to correlate distribution shift (entropy)
    with TTA benefit (adaptation_gain).

    Fields
    ------
    corruption_type : str
        Name of the corruption.
    severity : int
        Severity level (typically 3 for representative analysis).
    mean_entropy : float
        Mean Shannon entropy H(p) = -Σ p log p over all test samples.
        Unit: nats. Maximum = log(10) ≈ 2.303 for 10 classes (uniform).
    std_entropy : float
        Standard deviation of entropy across samples.
    mean_max_prob : float
        Mean of max-softmax probability. Low values → high uncertainty.
    fraction_uncertain : float
        Fraction of samples with max_prob < 0.5 (model not confident).
    baseline_accuracy : float
        Accuracy with no adaptation.
    tent_accuracy : float
        Accuracy after TENT adaptation.
    adaptation_gain : float
        tent_accuracy - baseline_accuracy. The "TTA benefit" variable
        used in the entropy–gain Pearson correlation.
    """

    corruption_type: str
    severity: int
    mean_entropy: float
    std_entropy: float
    mean_max_prob: float
    fraction_uncertain: float
    baseline_accuracy: float = 0.0
    tent_accuracy: float = 0.0
    adaptation_gain: float = 0.0

    def compute_adaptation_gain(self) -> float:
        """Calculate and store tent_accuracy - baseline_accuracy."""
        self.adaptation_gain = self.tent_accuracy - self.baseline_accuracy
        return self.adaptation_gain

    @property
    def normalized_entropy(self) -> float:
        """Entropy normalized to [0, 1] relative to maximum (log C)."""
        max_entropy = math.log(10)  # 10 CIFAR-10 classes
        return self.mean_entropy / max_entropy if max_entropy > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"UncertaintyMetrics({self.corruption_type}, "
            f"H={self.mean_entropy:.4f}, "
            f"gain={self.adaptation_gain:+.4f})"
        )


@dataclass
class BenchmarkSummary:
    """
    Top-level container for a complete benchmark run.

    Produced by BenchmarkEvaluator.finalize() and consumed by ReportGenerator.

    Fields
    ------
    methods : list[str]
        Methods evaluated (subset of METHODS).
    corruption_types : list[str]
        Corruption types evaluated (subset of ALL_CORRUPTIONS).
    severity : int
        Severity level(s) evaluated.
    mce_scores : dict
        {method: mCE} — mean corruption error (lower = better).
    relative_improvements : dict
        {method: relative mCE improvement over no_adaptation baseline}.
    accuracy_table : dict
        {corruption: {method: accuracy}}.
    entropy_table : dict
        {corruption: {method: entropy_before}}.
    winners : dict
        {corruption: winning_method}.
    profiles : dict
        {corruption: CorruptionProfile}.
    uncertainty_records : dict
        {corruption: UncertaintyMetrics}.
    pearson_r : float
        Pearson correlation between pre-adaptation entropy and TENT gain.
    pseudo_label_blur_failures : dict
        {corruption: {no_adaptation, pseudo_label, degradation}} for blur
        corruptions where pseudo-label underperformed the baseline.
    """

    methods: List[str] = field(default_factory=lambda: list(METHODS))
    corruption_types: List[str] = field(default_factory=list)
    severity: int = 3
    mce_scores: Dict[str, float] = field(default_factory=dict)
    relative_improvements: Dict[str, float] = field(default_factory=dict)
    accuracy_table: Dict[str, Dict[str, float]] = field(default_factory=dict)
    entropy_table: Dict[str, Dict[str, float]] = field(default_factory=dict)
    winners: Dict[str, str] = field(default_factory=dict)
    profiles: Dict[str, CorruptionProfile] = field(default_factory=dict)
    uncertainty_records: Dict[str, UncertaintyMetrics] = field(default_factory=dict)
    pearson_r: float = 0.0
    pseudo_label_blur_failures: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Computed properties                                                  #
    # ------------------------------------------------------------------ #

    @property
    def best_method(self) -> str:
        """Method with the lowest mCE (best overall performance)."""
        if not self.mce_scores:
            return ""
        return min(self.mce_scores, key=self.mce_scores.get)  # type: ignore[arg-type]

    @property
    def baseline_mce(self) -> float:
        """mCE of the no-adaptation baseline."""
        return self.mce_scores.get("no_adaptation", 0.0)

    @property
    def best_mce(self) -> float:
        """Lowest mCE across all methods."""
        return min(self.mce_scores.values()) if self.mce_scores else 0.0

    @property
    def best_improvement(self) -> float:
        """Largest relative mCE improvement vs. baseline."""
        return max(self.relative_improvements.values(), default=0.0)

    @property
    def n_corruptions(self) -> int:
        return len(self.corruption_types)

    def get_accuracy(self, corruption: str, method: str) -> Optional[float]:
        return self.accuracy_table.get(corruption, {}).get(method)

    def get_error_rate(self, corruption: str, method: str) -> Optional[float]:
        acc = self.get_accuracy(corruption, method)
        return (1.0 - acc) if acc is not None else None
