"""
Unit tests for entropy computation and the UncertaintyAnalyzer.

Test coverage:
    1. Shannon entropy is 0 for one-hot (certain) predictions.
    2. Shannon entropy equals log(C) for uniform predictions.
    3. Entropy is strictly between 0 and log(C) for intermediate distributions.
    4. pearson_correlation() returns ±1 for perfectly correlated/anti-correlated data.
    5. pearson_correlation() returns 0 for constant input (zero variance).
    6. UncertaintyAnalyzer.record() and compute_correlation() work end-to-end.
    7. compute_pre_adaptation_entropy() returns expected keys and valid ranges.
    8. UncertaintyAnalyzer.generate_ascii_scatter() produces non-empty output.

All tests run on CPU with synthetic data only.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn

from src.backbone.pretrained_model import CIFAR10ResNet
from src.uncertainty.uncertainty_analyzer import (
    UncertaintyAnalyzer,
    compute_pre_adaptation_entropy,
    pearson_correlation,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tiny_model() -> nn.Module:
    """Return a tiny model with BN layers for entropy tests."""

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1, bias=False),
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(8, 10),
            )

        def forward(self, x):
            return self.net(x)

    model = TinyNet()
    model.eval()
    return model


def _make_loader(n: int = 64, seed: int = 99):
    torch.manual_seed(seed)
    images = torch.randn(n, 3, 32, 32)
    labels = torch.randint(0, 10, (n,))
    ds     = torch.utils.data.TensorDataset(images, labels)
    return torch.utils.data.DataLoader(ds, batch_size=16)


# ---------------------------------------------------------------------------
# Tests: Shannon entropy properties
# ---------------------------------------------------------------------------


class TestShannonEntropy:
    """Tests for the static CIFAR10ResNet.get_entropy() method."""

    NUM_CLASSES = 10

    def test_zero_for_one_hot(self):
        """
        Perfectly confident prediction → entropy = 0.

        A logit of +∞ on class 0 (approximated by a large value) should
        produce a softmax of [1, 0, …, 0] and entropy ≈ 0.
        """
        logits = torch.zeros(1, self.NUM_CLASSES)
        logits[0, 0] = 1e6   # effectively one-hot
        entropy = CIFAR10ResNet.get_entropy(logits)
        assert entropy.shape == (1,)
        assert entropy.item() < 1e-4, (
            f"Entropy should be ≈0 for one-hot, got {entropy.item():.6f}"
        )

    def test_max_for_uniform(self):
        """
        Uniform distribution → maximum entropy = log(C).

        Zero logits → softmax is exactly uniform over C classes.
        """
        logits         = torch.zeros(1, self.NUM_CLASSES)
        entropy        = CIFAR10ResNet.get_entropy(logits)
        expected_max   = math.log(self.NUM_CLASSES)  # log(10) ≈ 2.3026

        assert abs(entropy.item() - expected_max) < 1e-3, (
            f"Entropy should be log({self.NUM_CLASSES})={expected_max:.4f}, "
            f"got {entropy.item():.4f}"
        )

    def test_between_zero_and_max_for_intermediate(self):
        """Intermediate distributions have 0 < H < log(C)."""
        logits  = torch.tensor([[2.0, 1.0, 0.5] + [0.0] * 7])
        entropy = CIFAR10ResNet.get_entropy(logits).item()
        assert 0.0 < entropy < math.log(self.NUM_CLASSES), (
            f"Entropy {entropy:.4f} not in (0, {math.log(self.NUM_CLASSES):.4f})"
        )

    def test_batch_shape(self):
        """Output shape matches batch size."""
        batch_size = 32
        logits     = torch.randn(batch_size, self.NUM_CLASSES)
        entropy    = CIFAR10ResNet.get_entropy(logits)
        assert entropy.shape == (batch_size,)

    def test_non_negative(self):
        """Entropy is always ≥ 0."""
        for _ in range(20):
            logits  = torch.randn(16, self.NUM_CLASSES)
            entropy = CIFAR10ResNet.get_entropy(logits)
            assert (entropy >= -1e-6).all(), "Entropy should be non-negative"

    def test_monotone_with_peakedness(self):
        """
        A more peaked distribution (higher logit on class 0) should have
        lower entropy than a flatter one.
        """
        logits_flat   = torch.zeros(1, self.NUM_CLASSES)
        logits_peaked = torch.zeros(1, self.NUM_CLASSES)
        logits_peaked[0, 0] = 5.0

        h_flat   = CIFAR10ResNet.get_entropy(logits_flat).item()
        h_peaked = CIFAR10ResNet.get_entropy(logits_peaked).item()

        assert h_peaked < h_flat, (
            f"Peaked distribution should have lower entropy: "
            f"{h_peaked:.4f} vs {h_flat:.4f}"
        )

    def test_symmetry(self):
        """
        Permuting the logits should not change entropy
        (entropy depends only on the probability values, not their order).
        """
        logits       = torch.tensor([[3.0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        logits_perm  = torch.tensor([[0.0, 3.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0]])

        h1 = CIFAR10ResNet.get_entropy(logits).item()
        h2 = CIFAR10ResNet.get_entropy(logits_perm).item()
        assert abs(h1 - h2) < 1e-5, f"Permuted logits gave different entropy: {h1:.6f} vs {h2:.6f}"

    def test_consistent_with_manual_calculation(self):
        """Compare against a manual numpy-style entropy calculation."""
        import math as _math
        probs  = [0.7, 0.2, 0.1] + [0.0] * 7
        manual = -sum(p * _math.log(p + 1e-8) for p in probs)

        # Make logits whose softmax ≈ probs (use log(p) as logit)
        logits = torch.log(torch.tensor(probs) + 1e-8).unsqueeze(0)
        h      = CIFAR10ResNet.get_entropy(logits).item()

        assert abs(h - manual) < 0.05, (
            f"Entropy {h:.4f} diverges from manual {manual:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: Pearson correlation
# ---------------------------------------------------------------------------


class TestPearsonCorrelation:
    """Tests for the pure-Python pearson_correlation function."""

    def test_perfect_positive_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [2.0, 4.0, 6.0, 8.0, 10.0]
        r = pearson_correlation(x, y)
        assert abs(r - 1.0) < 1e-9, f"Expected r=1.0, got {r}"

    def test_perfect_negative_correlation(self):
        x = [1.0, 2.0, 3.0, 4.0, 5.0]
        y = [5.0, 4.0, 3.0, 2.0, 1.0]
        r = pearson_correlation(x, y)
        assert abs(r - (-1.0)) < 1e-9, f"Expected r=-1.0, got {r}"

    def test_zero_correlation_orthogonal(self):
        """
        Data where x and y are truly orthogonal (dot product = 0 after mean
        subtraction) should produce r = 0.
        """
        # x - mean(x) = [-1.5, -0.5, 0.5, 1.5]
        # y - mean(y) = [-1,    1,   -1,    1]
        # dot product = (-1.5)(-1) + (-0.5)(1) + (0.5)(-1) + (1.5)(1)
        #             = 1.5 - 0.5 - 0.5 + 1.5 = 2.0   (not zero for these values)
        # Use: x = [1,2,3,4], y constructed so cov = 0
        # mean(x) = 2.5, mean(y) = 2.5
        # cov = (1-2.5)(4-2.5) + (2-2.5)(1-2.5) + (3-2.5)(4-2.5) + (4-2.5)(1-2.5)
        #     = (-1.5)(1.5) + (-0.5)(-1.5) + (0.5)(1.5) + (1.5)(-1.5)
        #     = -2.25 + 0.75 + 0.75 - 2.25 = -3.0   (still not zero)
        # Correct truly uncorrelated example (analytical zero covariance):
        # x = [1,2,3,4], y = [2,4,2,4]  → cov ≠ 0
        # Use a known-zero-covariance construction:
        # Let x = [-1, 0, 1], y = [1, -2, 1]  (y = x^2 - 2/3, orthogonal to x over {-1,0,1})
        x = [-1.0, 0.0, 1.0]
        y = [1.0, -2.0, 1.0]
        r = pearson_correlation(x, y)
        assert abs(r) < 1e-9, f"Expected r=0 for orthogonal data, got {r}"

    def test_constant_x_returns_zero(self):
        """Zero variance in x → return 0 (undefined correlation)."""
        x = [3.0, 3.0, 3.0, 3.0]
        y = [1.0, 2.0, 3.0, 4.0]
        r = pearson_correlation(x, y)
        assert r == 0.0

    def test_constant_y_returns_zero(self):
        x = [1.0, 2.0, 3.0, 4.0]
        y = [5.0, 5.0, 5.0, 5.0]
        r = pearson_correlation(x, y)
        assert r == 0.0

    def test_range_is_minus_one_to_one(self):
        """Pearson r must be in [-1, 1] for any inputs."""
        import random
        rng = random.Random(42)
        for _ in range(50):
            x = [rng.gauss(0, 1) for _ in range(20)]
            y = [rng.gauss(0, 1) for _ in range(20)]
            r = pearson_correlation(x, y)
            assert -1.0 - 1e-9 <= r <= 1.0 + 1e-9, f"r={r} out of [-1,1]"

    def test_mismatched_lengths_returns_zero(self):
        r = pearson_correlation([1.0, 2.0], [1.0, 2.0, 3.0])
        assert r == 0.0

    def test_fewer_than_two_points_returns_zero(self):
        assert pearson_correlation([], []) == 0.0
        assert pearson_correlation([1.0], [1.0]) == 0.0


# ---------------------------------------------------------------------------
# Tests: compute_pre_adaptation_entropy
# ---------------------------------------------------------------------------


class TestComputePreAdaptationEntropy:
    """Tests for the standalone entropy computation function."""

    def test_returns_expected_keys(self):
        model   = _make_tiny_model()
        loader  = _make_loader()
        metrics = compute_pre_adaptation_entropy(model, loader)

        for key in ("mean_entropy", "std_entropy", "mean_max_prob",
                    "fraction_uncertain", "num_samples"):
            assert key in metrics, f"Missing key: {key}"

    def test_entropy_in_valid_range(self):
        model   = _make_tiny_model()
        loader  = _make_loader()
        metrics = compute_pre_adaptation_entropy(model, loader)

        assert 0.0 <= metrics["mean_entropy"] <= math.log(10) + 0.01, (
            f"mean_entropy={metrics['mean_entropy']:.4f} out of [0, log(10)]"
        )

    def test_max_prob_in_valid_range(self):
        model   = _make_tiny_model()
        loader  = _make_loader()
        metrics = compute_pre_adaptation_entropy(model, loader)

        assert 0.0 <= metrics["mean_max_prob"] <= 1.0 + 1e-6, (
            f"mean_max_prob={metrics['mean_max_prob']:.4f} out of [0,1]"
        )

    def test_fraction_uncertain_in_valid_range(self):
        model   = _make_tiny_model()
        loader  = _make_loader()
        metrics = compute_pre_adaptation_entropy(model, loader)

        assert 0.0 <= metrics["fraction_uncertain"] <= 1.0 + 1e-6

    def test_num_samples_matches_dataset(self):
        n       = 48
        model   = _make_tiny_model()
        images  = torch.randn(n, 3, 32, 32)
        labels  = torch.randint(0, 10, (n,))
        loader  = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(images, labels), batch_size=16
        )
        metrics = compute_pre_adaptation_entropy(model, loader)
        assert metrics["num_samples"] == n

    def test_certain_model_low_entropy(self):
        """A model with constant high-logit output should produce low entropy."""
        class CertainModel(nn.Module):
            def forward(self, x):
                out = torch.zeros(x.size(0), 10)
                out[:, 0] = 100.0   # always predicts class 0 with certainty
                return out

        loader  = _make_loader(n=32)
        metrics = compute_pre_adaptation_entropy(CertainModel(), loader)
        assert metrics["mean_entropy"] < 0.01, (
            f"Certain model should have near-zero entropy, got {metrics['mean_entropy']:.4f}"
        )

    def test_uniform_model_max_entropy(self):
        """A model outputting uniform logits should give entropy ≈ log(10)."""
        class UniformModel(nn.Module):
            def forward(self, x):
                return torch.zeros(x.size(0), 10)   # equal logits → uniform softmax

        loader  = _make_loader(n=32)
        metrics = compute_pre_adaptation_entropy(UniformModel(), loader)
        assert abs(metrics["mean_entropy"] - math.log(10)) < 0.01, (
            f"Uniform model should have max entropy {math.log(10):.4f}, "
            f"got {metrics['mean_entropy']:.4f}"
        )


# ---------------------------------------------------------------------------
# Tests: UncertaintyAnalyzer end-to-end
# ---------------------------------------------------------------------------


class TestUncertaintyAnalyzer:
    """Integration tests for the UncertaintyAnalyzer class."""

    def _build_analyzer(self) -> UncertaintyAnalyzer:
        return UncertaintyAnalyzer(_make_tiny_model())

    def test_record_and_correlation_with_two_points(self):
        """With 2 corruptions, correlation should be computable."""
        analyzer = self._build_analyzer()
        analyzer.record(
            "gaussian_noise",
            {"mean_entropy": 1.8, "std_entropy": 0.2, "mean_max_prob": 0.3,
             "fraction_uncertain": 0.6, "num_samples": 100},
            baseline_accuracy=0.40,
            tent_accuracy=0.55,
        )
        analyzer.record(
            "brightness",
            {"mean_entropy": 0.8, "std_entropy": 0.1, "mean_max_prob": 0.7,
             "fraction_uncertain": 0.1, "num_samples": 100},
            baseline_accuracy=0.70,
            tent_accuracy=0.72,
        )
        r = analyzer.compute_correlation()
        # gaussian_noise: high entropy (1.8), high gain (0.15)
        # brightness:     low entropy (0.8),  low gain (0.02)
        # → positive correlation expected
        assert r > 0.0, f"Expected positive r, got {r}"

    def test_fewer_than_two_records_returns_zero(self):
        analyzer = self._build_analyzer()
        assert analyzer.compute_correlation() == 0.0
        analyzer.record(
            "gaussian_noise",
            {"mean_entropy": 1.0, "std_entropy": 0.1, "mean_max_prob": 0.5,
             "fraction_uncertain": 0.3, "num_samples": 50},
            baseline_accuracy=0.5,
            tent_accuracy=0.6,
        )
        assert analyzer.compute_correlation() == 0.0

    def test_ranked_corruptions_descending_entropy(self):
        analyzer = self._build_analyzer()
        analyzer.record(
            "low_entropy_corruption",
            {"mean_entropy": 0.5, "std_entropy": 0.05, "mean_max_prob": 0.85,
             "fraction_uncertain": 0.05, "num_samples": 100},
            baseline_accuracy=0.80,
            tent_accuracy=0.81,
        )
        analyzer.record(
            "high_entropy_corruption",
            {"mean_entropy": 2.0, "std_entropy": 0.20, "mean_max_prob": 0.20,
             "fraction_uncertain": 0.70, "num_samples": 100},
            baseline_accuracy=0.30,
            tent_accuracy=0.45,
        )
        ranked = analyzer.get_ranked_corruptions()
        assert ranked[0][0] == "high_entropy_corruption"
        assert ranked[1][0] == "low_entropy_corruption"

    def test_generate_report_returns_string(self):
        analyzer = self._build_analyzer()
        analyzer.record(
            "c1",
            {"mean_entropy": 1.0, "std_entropy": 0.1, "mean_max_prob": 0.6,
             "fraction_uncertain": 0.2, "num_samples": 50},
            baseline_accuracy=0.5, tent_accuracy=0.6,
        )
        analyzer.record(
            "c2",
            {"mean_entropy": 1.5, "std_entropy": 0.15, "mean_max_prob": 0.4,
             "fraction_uncertain": 0.4, "num_samples": 50},
            baseline_accuracy=0.4, tent_accuracy=0.55,
        )
        report = analyzer.generate_report()
        assert isinstance(report, str)
        assert "Pearson" in report
        assert "c1" in report
        assert "c2" in report

    def test_generate_ascii_scatter_non_empty(self):
        analyzer = self._build_analyzer()
        for i, name in enumerate(["c1", "c2", "c3"]):
            analyzer.record(
                name,
                {"mean_entropy": 0.5 + i * 0.5, "std_entropy": 0.1,
                 "mean_max_prob": 0.8 - i * 0.2, "fraction_uncertain": i * 0.1,
                 "num_samples": 50},
                baseline_accuracy=0.6 - i * 0.1,
                tent_accuracy=0.65 - i * 0.05,
            )
        scatter = analyzer.generate_ascii_scatter()
        assert isinstance(scatter, str)
        assert len(scatter) > 10

    def test_compute_entropy_uses_model(self):
        """compute_entropy() should return metrics based on model outputs."""

        class FixedModel(nn.Module):
            """Always outputs uniform logits → max entropy."""
            def forward(self, x):
                return torch.zeros(x.size(0), 10)

        analyzer = UncertaintyAnalyzer(FixedModel())
        loader   = _make_loader(n=32)
        metrics  = analyzer.compute_entropy(loader)

        assert abs(metrics["mean_entropy"] - math.log(10)) < 0.01, (
            f"Expected max entropy {math.log(10):.4f}, got {metrics['mean_entropy']:.4f}"
        )
