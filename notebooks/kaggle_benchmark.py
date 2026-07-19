#!/usr/bin/env python3
"""
Domain Adaptation Benchmark — Kaggle Notebook
===============================================
Complete, self-contained benchmark for test-time adaptation methods.
Designed to run end-to-end on Kaggle with GPU acceleration.

Run this as a Kaggle notebook or locally:
    python notebooks/kaggle_benchmark.py

What this notebook does:
    1. Downloads CIFAR-10 (for fine-tuning) and CIFAR-10-C (for evaluation)
    2. Fine-tunes ResNet-50 backbone on clean CIFAR-10
    3. Evaluates 4 TTA methods on 15 corruption types
    4. Generates publication-quality figures and statistical analysis
    5. Produces an HTML report with all results

Estimated runtime:
    - Kaggle GPU (T4):  ~15 minutes
    - Local CPU:        ~45 minutes

Author: Adnan Hassnain | BS CS, NUST Pakistan
Repo:   https://github.com/adnaan512/domain-adaptation-benchmark
"""

from __future__ import annotations

import logging
import os
import sys
import time
import subprocess

# ── Setup paths ──────────────────────────────────────────────────────────────

# Detect if running on Kaggle
IS_KAGGLE = os.path.exists("/kaggle")

if IS_KAGGLE:
    PROJECT_ROOT = "/kaggle/working/domain-adaptation-benchmark"
    # Clone the repo if not already present
    if not os.path.exists(PROJECT_ROOT):
        subprocess.run(
            ["git", "clone",
             "https://github.com/adnaan512/domain-adaptation-benchmark.git",
             PROJECT_ROOT],
            check=True,
        )
    sys.path.insert(0, PROJECT_ROOT)
    os.chdir(PROJECT_ROOT)
else:
    # Local run — assume we're in the project root or notebooks/ dir
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)
    os.chdir(project_root)

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Imports (after path setup) ───────────────────────────────────────────────

import torch
import numpy as np

from src.backbone.pretrained_model import build_model, fine_tune_on_cifar10
from src.data.dataset_loader import (
    CIFAR10CLoader,
    MockCorruptionLoader,
    get_cifar10_loaders,
    CORRUPTION_TYPES,
    MOCK_CORRUPTION_TYPES,
)
from src.adaptation.no_adaptation import evaluate_no_adaptation
from src.adaptation.test_time_norm import adapt_with_ttn
from src.adaptation.tent import adapt_with_tent
from src.adaptation.pseudo_label import adapt_with_pseudo_label
from src.benchmark.evaluator import BenchmarkEvaluator
from src.benchmark.stats import compute_all_stats, format_stats_table
from src.uncertainty.uncertainty_analyzer import UncertaintyAnalyzer
from src.reporting.report_generator import ReportGenerator

# Lazy import matplotlib (may not be available in all environments)
try:
    from src.reporting.visualize import generate_all_figures
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available — skipping figure generation.")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

class Config:
    """Centralised configuration for the benchmark run."""

    # ── Paths ────────────────────────────────────────────────────────────
    CIFAR10_DIR = "./data"                 # Clean CIFAR-10 download location
    CIFAR10C_DIR = "./CIFAR-10-C"          # Corrupted CIFAR-10-C location
    WEIGHTS_PATH = "./cifar10_resnet50.pth"  # Fine-tuned checkpoint
    FIGURES_DIR = "./figures"               # Output figures directory
    REPORT_PATH = "./benchmark_report.html" # HTML report output

    # ── Fine-tuning ──────────────────────────────────────────────────────
    FT_EPOCHS = 10         # Fine-tuning epochs (10 is sufficient)
    FT_LR = 0.01           # SGD learning rate
    FT_BATCH_SIZE = 128    # Training batch size

    # ── Benchmark ────────────────────────────────────────────────────────
    SEVERITY = 3           # Default severity level
    BATCH_SIZE = 128       # Evaluation batch size
    TENT_LR = 1e-3         # TENT Adam learning rate
    PL_THRESHOLD = 0.9     # Pseudo-label confidence threshold

    # ── Device ───────────────────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0: ENVIRONMENT CHECK
# ══════════════════════════════════════════════════════════════════════════════

def check_environment():
    """Print system information and verify dependencies."""
    print("=" * 70)
    print("  DOMAIN ADAPTATION BENCHMARK — Environment Check")
    print("=" * 70)
    print(f"  Python:     {sys.version.split()[0]}")
    print(f"  PyTorch:    {torch.__version__}")
    print(f"  NumPy:      {np.__version__}")
    print(f"  Device:     {Config.DEVICE}")
    if Config.DEVICE.type == "cuda":
        print(f"  GPU:        {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU Memory: {mem:.1f} GB")
    print(f"  Kaggle:     {'Yes' if IS_KAGGLE else 'No'}")
    print(f"  matplotlib: {'Available' if HAS_MATPLOTLIB else 'Not available'}")
    print("=" * 70)
    print()


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD CIFAR-10-C (if not present)
# ══════════════════════════════════════════════════════════════════════════════

def download_cifar10c():
    """
    Download and extract CIFAR-10-C from Zenodo if not already present.

    The dataset is ~300 MB compressed, ~800 MB extracted.
    Contains 15 .npy files (one per corruption type) + labels.npy.
    """
    if os.path.isdir(Config.CIFAR10C_DIR):
        npy_count = len([
            f for f in os.listdir(Config.CIFAR10C_DIR)
            if f.endswith(".npy")
        ])
        if npy_count >= 16:  # 15 corruptions + labels
            logger.info(
                "CIFAR-10-C already present (%d .npy files). Skipping download.",
                npy_count,
            )
            return True

    logger.info("Downloading CIFAR-10-C from Zenodo (~300 MB)...")
    url = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
    tar_path = "./CIFAR-10-C.tar"

    try:
        # Try wget first (usually available on Linux/Kaggle)
        subprocess.run(
            ["wget", "-q", "--show-progress", url, "-O", tar_path],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to Python download
        logger.info("wget not available, using Python urllib...")
        import urllib.request
        urllib.request.urlretrieve(url, tar_path)

    logger.info("Extracting CIFAR-10-C.tar...")
    import tarfile
    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(".")

    # Clean up
    if os.path.exists(tar_path):
        os.remove(tar_path)

    logger.info("CIFAR-10-C ready at %s", Config.CIFAR10C_DIR)
    return True


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2: FINE-TUNE BACKBONE ON CLEAN CIFAR-10
# ══════════════════════════════════════════════════════════════════════════════

def fine_tune_backbone():
    """
    Fine-tune ResNet-50 on clean CIFAR-10 (or load existing checkpoint).

    Only layer4, FC head, and BN affine parameters are updated.
    Earlier backbone layers retain ImageNet features.

    Returns
    -------
    CIFAR10ResNet
        Fine-tuned model ready for benchmark evaluation.
    """
    device = Config.DEVICE

    if os.path.exists(Config.WEIGHTS_PATH):
        logger.info(
            "Found existing checkpoint: %s — loading.",
            Config.WEIGHTS_PATH,
        )
        model = build_model(
            weights_path=Config.WEIGHTS_PATH, device=device,
        )
        return model

    logger.info("No checkpoint found. Fine-tuning on clean CIFAR-10...")
    logger.info("  Epochs:     %d", Config.FT_EPOCHS)
    logger.info("  LR:         %g", Config.FT_LR)
    logger.info("  Batch size: %d", Config.FT_BATCH_SIZE)

    # Download and load CIFAR-10
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=Config.CIFAR10_DIR, batch_size=Config.FT_BATCH_SIZE,
    )

    model = build_model(device=device)

    t0 = time.perf_counter()
    fine_tune_on_cifar10(
        model, train_loader, device=device,
        epochs=Config.FT_EPOCHS, lr=Config.FT_LR,
        save_path=Config.WEIGHTS_PATH,
    )
    elapsed = time.perf_counter() - t0
    logger.info("Fine-tuning completed in %.1f minutes.", elapsed / 60)

    # Evaluate on clean test set
    from src.backbone.pretrained_model import evaluate_on_clean
    clean_acc = evaluate_on_clean(model, test_loader, device)
    logger.info("Clean CIFAR-10 test accuracy: %.2f%%", 100 * clean_acc)

    return model


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3: RUN BENCHMARK
# ══════════════════════════════════════════════════════════════════════════════

METHODS = {
    "no_adaptation":  evaluate_no_adaptation,
    "test_time_norm": adapt_with_ttn,
    "tent":           adapt_with_tent,
    "pseudo_label":   adapt_with_pseudo_label,
}

METHOD_DISPLAY = {
    "no_adaptation":  "No Adapt",
    "test_time_norm": "TTN",
    "tent":           "TENT",
    "pseudo_label":   "Pseudo-Label",
}


def run_benchmark(model, use_real_data: bool = True):
    """
    Run the full benchmark: 4 methods × all available corruptions.

    Parameters
    ----------
    model : CIFAR10ResNet
        Fine-tuned model.
    use_real_data : bool
        If True, uses CIFAR-10-C. If False, uses mock data (for testing).

    Returns
    -------
    tuple of (BenchmarkSummary, UncertaintyAnalyzer, dict)
        Summary, analyzer, and raw results.
    """
    device = Config.DEVICE
    severity = Config.SEVERITY

    if use_real_data and os.path.isdir(Config.CIFAR10C_DIR):
        loader_obj = CIFAR10CLoader(
            Config.CIFAR10C_DIR, batch_size=Config.BATCH_SIZE,
        )
        corruptions = loader_obj.available_corruptions()
        get_loader = loader_obj.get_loader
        logger.info("Using real CIFAR-10-C data (%d corruptions).", len(corruptions))
    else:
        mock = MockCorruptionLoader(
            batch_size=Config.BATCH_SIZE, num_samples=640,
        )
        corruptions = MOCK_CORRUPTION_TYPES
        get_loader = mock.get_loader
        logger.info("Using mock data (%d corruptions).", len(corruptions))

    evaluator = BenchmarkEvaluator(corruptions, severity=severity)
    uncertainty_analyzer = UncertaintyAnalyzer(model, device)
    all_results = {}

    t_total = time.perf_counter()

    for idx, corruption in enumerate(corruptions, 1):
        print()
        logger.info(
            "[%d/%d]  %s  (severity %d)",
            idx, len(corruptions), corruption, severity,
        )
        all_results[corruption] = {}

        for method_name, method_fn in METHODS.items():
            # Reset model to original state before each method
            model.restore_original_state()
            model.eval()

            loader = get_loader(corruption, severity)
            t0 = time.perf_counter()

            if method_name == "tent":
                result = method_fn(
                    model, loader, device, lr=Config.TENT_LR,
                )
            elif method_name == "pseudo_label":
                result = method_fn(
                    model, loader, device,
                    confidence_threshold=Config.PL_THRESHOLD,
                )
            else:
                result = method_fn(model, loader, device)

            elapsed = time.perf_counter() - t0
            acc = result["accuracy"]
            h_before = result.get(
                "mean_entropy", result.get("mean_entropy_before", 0.0),
            )

            evaluator.add_result(corruption, method_name, acc, h_before)
            all_results[corruption][method_name] = result

            logger.info(
                "  %-20s  acc=%6.2f%%  H=%.4f  [%.1fs]",
                METHOD_DISPLAY.get(method_name, method_name),
                100 * acc, h_before, elapsed,
            )

        # Uncertainty analysis
        model.restore_original_state()
        model.eval()
        ua_loader = get_loader(corruption, severity)
        entropy_metrics = uncertainty_analyzer.compute_entropy(ua_loader)
        uncertainty_analyzer.record(
            corruption, entropy_metrics,
            baseline_accuracy=all_results[corruption]["no_adaptation"]["accuracy"],
            tent_accuracy=all_results[corruption]["tent"]["accuracy"],
        )

    total_time = time.perf_counter() - t_total
    logger.info(
        "Benchmark completed in %.1f minutes (%d corruptions x 4 methods).",
        total_time / 60, len(corruptions),
    )

    # Finalise
    summary = evaluator.finalize()
    pearson_r = uncertainty_analyzer.compute_correlation()
    summary.pearson_r = pearson_r

    # Print reports
    print()
    print(evaluator.generate_full_report())
    print()
    print(uncertainty_analyzer.generate_report())

    return summary, uncertainty_analyzer, all_results


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4: STATISTICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def run_statistical_analysis(summary):
    """
    Compute and print statistical analysis with confidence intervals.

    Parameters
    ----------
    summary : BenchmarkSummary
        Completed benchmark results.

    Returns
    -------
    dict
        Per-method statistical results.
    """
    print()
    print("=" * 85)
    print("  STATISTICAL ANALYSIS")
    print("=" * 85)

    stats = compute_all_stats(summary)
    print()
    print(format_stats_table(stats))
    print()

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5: GENERATE FIGURES
# ══════════════════════════════════════════════════════════════════════════════

def generate_figures(summary, uncertainty_analyzer):
    """
    Generate publication-quality matplotlib figures.

    Parameters
    ----------
    summary : BenchmarkSummary
        Completed benchmark results.
    uncertainty_analyzer : UncertaintyAnalyzer
        For entropy correlation analysis.

    Returns
    -------
    list of str
        Paths to generated figure files.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not available — skipping figures.")
        return []

    logger.info("Generating publication-quality figures...")
    os.makedirs(Config.FIGURES_DIR, exist_ok=True)

    saved = generate_all_figures(
        summary,
        uncertainty_analyzer=uncertainty_analyzer,
        output_dir=Config.FIGURES_DIR,
    )

    logger.info("Generated %d figures in %s/", len(saved), Config.FIGURES_DIR)
    return saved


# ══════════════════════════════════════════════════════════════════════════════
# STEP 6: GENERATE HTML REPORT
# ══════════════════════════════════════════════════════════════════════════════

def generate_report(summary, uncertainty_analyzer):
    """
    Generate the self-contained dark-theme HTML report.

    Parameters
    ----------
    summary : BenchmarkSummary
        Completed benchmark results.
    uncertainty_analyzer : UncertaintyAnalyzer
        For entropy correlation analysis.
    """
    logger.info("Generating HTML report...")
    report_gen = ReportGenerator(
        summary,
        uncertainty_analyzer=uncertainty_analyzer,
        pearson_r=summary.pearson_r,
        title="Domain Adaptation Benchmark — Full Results",
    )
    html_content = report_gen.generate()

    with open(Config.REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write(html_content)

    logger.info("HTML report saved -> %s", os.path.abspath(Config.REPORT_PATH))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """Run the complete benchmark pipeline end-to-end."""
    t_start = time.perf_counter()

    print()
    print("+" + "=" * 68 + "+")
    print("|  DOMAIN ADAPTATION BENCHMARK                                       |")
    print("|  Test-Time Adaptation for Distribution Shift in Deep Learning      |")
    print("|  Author: Adnan Hassnain | BS CS, NUST Pakistan                     |")
    print("+" + "=" * 68 + "+")
    print()

    # Step 0: Environment check
    check_environment()

    # Step 1: Download CIFAR-10-C
    has_real_data = download_cifar10c()

    # Step 2: Fine-tune backbone
    model = fine_tune_backbone()

    # Step 3: Run benchmark
    summary, uncertainty_analyzer, raw_results = run_benchmark(
        model, use_real_data=has_real_data,
    )

    # Step 4: Statistical analysis
    stats = run_statistical_analysis(summary)

    # Step 5: Generate figures
    figure_paths = generate_figures(summary, uncertainty_analyzer)

    # Step 6: Generate HTML report
    generate_report(summary, uncertainty_analyzer)

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = time.perf_counter() - t_start
    print()
    print("+" + "=" * 68 + "+")
    print("|  BENCHMARK COMPLETE                                                |")
    print("+" + "=" * 68 + "+")
    print(f"|  Total time:    {total_time / 60:.1f} minutes")
    print(f"|  Corruptions:   {len(summary.corruption_types)}")
    print(f"|  Best method:   {summary.best_method} "
          f"(mCE = {summary.best_mce:.4f})")
    print(f"|  Pearson r:     {summary.pearson_r:+.4f}")
    print(f"|  Figures:       {len(figure_paths)} generated")
    print(f"|  HTML report:   {Config.REPORT_PATH}")
    print("+" + "=" * 68 + "+")
    print()

    if IS_KAGGLE:
        print("  Figures and report are saved in /kaggle/working/")
        print("  Download them from the 'Output' tab.")
    else:
        print(f"  Open the report: file://{os.path.abspath(Config.REPORT_PATH)}")
        print(f"  View figures in: {os.path.abspath(Config.FIGURES_DIR)}/")
    print()


if __name__ == "__main__":
    main()
