#!/usr/bin/env python3
"""
Domain Adaptation Benchmark — CLI Entry Point
==============================================
Systematic evaluation of test-time adaptation methods for distribution shift.

Usage
-----
Demo (no download, ~30 s):
    python main.py --mode demo

Full benchmark (requires CIFAR-10-C download, ~30 min on CPU):
    python main.py --mode full --data-dir ./CIFAR-10-C

Single corruption:
    python main.py --corruption gaussian_noise --severity 3 --data-dir ./CIFAR-10-C

Dry run (validate setup only):
    python main.py --dry-run

Fine-tune backbone on CIFAR-10 (one-time, ~2 h on CPU):
    python main.py --fine-tune --data-dir ./data

Author  : Adnan Hassnain | BS CS, NUST Pakistan
Repo    : https://github.com/adnaan512/domain-adaptation-benchmark
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import torch

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Project imports ───────────────────────────────────────────────────────────
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
from src.uncertainty.uncertainty_analyzer import UncertaintyAnalyzer
from src.reporting.report_generator import ReportGenerator

# ── Constants ─────────────────────────────────────────────────────────────────
METHODS = {
    "no_adaptation": evaluate_no_adaptation,
    "test_time_norm": adapt_with_ttn,
    "tent":           adapt_with_tent,
    "pseudo_label":   adapt_with_pseudo_label,
}

METHOD_DISPLAY = {
    "no_adaptation": "No Adapt",
    "test_time_norm": "TTN",
    "tent":           "TENT",
    "pseudo_label":   "Pseudo-Label",
}


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ---- Mode ----
    mode_group = p.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--mode",
        choices=["demo", "full", "full-sweep"],
        default=None,
        help=(
            "demo: synthetic data, no download. "
            "full: real CIFAR-10-C (requires --data-dir). "
            "full-sweep: all 5 severity levels (comprehensive)."
        ),
    )
    mode_group.add_argument(
        "--corruption",
        choices=CORRUPTION_TYPES,
        default=None,
        help="Evaluate a single corruption type (requires --data-dir).",
    )
    mode_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate model loading and data setup only; no adaptation.",
    )
    mode_group.add_argument(
        "--fine-tune",
        action="store_true",
        help="Fine-tune backbone on CIFAR-10 and save checkpoint.",
    )

    # ---- Data ----
    p.add_argument(
        "--data-dir",
        default="./CIFAR-10-C",
        help="Directory containing extracted CIFAR-10-C .npy files (mode=full).",
    )
    p.add_argument(
        "--cifar10-dir",
        default="./data",
        help="Directory for CIFAR-10 (auto-downloaded if not present).",
    )
    p.add_argument(
        "--severity",
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5],
        help="Corruption severity level.",
    )

    # ---- Model ----
    p.add_argument(
        "--weights-path",
        default="./cifar10_resnet50.pth",
        help="Path to fine-tuned CIFAR-10 checkpoint.",
    )

    # ---- Adaptation hyper-params ----
    p.add_argument("--tent-lr",      type=float, default=1e-3,
                   help="TENT Adam learning rate.")
    p.add_argument("--pl-threshold", type=float, default=0.9,
                   help="Pseudo-label confidence threshold.")
    p.add_argument("--batch-size",   type=int,   default=128,
                   help="Batch size for data loaders.")

    # ---- Fine-tune ----
    p.add_argument("--ft-epochs",    type=int,   default=10,
                   help="Epochs for fine-tuning (--fine-tune only).")
    p.add_argument("--ft-lr",        type=float, default=0.01,
                   help="SGD learning rate for fine-tuning.")

    # ---- Output ----
    p.add_argument(
        "--output",
        default="benchmark_report.html",
        help="Path for the generated HTML report.",
    )
    p.add_argument("--figures-dir", default="./figures",
                   help="Directory for publication-quality figures.")
    p.add_argument("--no-report", action="store_true",
                   help="Skip HTML report generation.")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation.")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Enable DEBUG-level logging.")

    return p


# ── Runner helpers ────────────────────────────────────────────────────────────

def _run_single_corruption(
    model,
    loader_fn,
    corruption: str,
    severity: int,
    device: torch.device,
    evaluator: BenchmarkEvaluator,
    uncertainty_analyzer: UncertaintyAnalyzer,
    tent_lr: float,
    pl_threshold: float,
) -> dict:
    """Run all 4 methods on one corruption type and record results."""
    results: dict = {}

    logger.info("━━━  %s  (severity %d)  ━━━", corruption, severity)

    for method_name, method_fn in METHODS.items():
        model.restore_original_state()
        model.eval()

        loader = loader_fn(corruption, severity)
        t0     = time.perf_counter()

        if method_name == "tent":
            result = method_fn(model, loader, device, lr=tent_lr)
        elif method_name == "pseudo_label":
            result = method_fn(model, loader, device,
                               confidence_threshold=pl_threshold)
        else:
            result = method_fn(model, loader, device)

        elapsed = time.perf_counter() - t0
        acc     = result["accuracy"]
        h_before = result.get("mean_entropy",
                   result.get("mean_entropy_before", 0.0))

        evaluator.add_result(corruption, method_name, acc, h_before)
        results[method_name] = result

        logger.info(
            "  %-20s  acc=%6.2f%%  H=%.4f  [%.1fs]",
            METHOD_DISPLAY.get(method_name, method_name),
            100 * acc, h_before, elapsed,
        )

    # Uncertainty: use no-adaptation pre-entropy
    model.restore_original_state()
    model.eval()
    ua_loader       = loader_fn(corruption, severity)
    entropy_metrics = uncertainty_analyzer.compute_entropy(ua_loader)
    uncertainty_analyzer.record(
        corruption,
        entropy_metrics,
        baseline_accuracy=results["no_adaptation"]["accuracy"],
        tent_accuracy=results["tent"]["accuracy"],
    )

    return results


def _finish(
    evaluator: BenchmarkEvaluator,
    uncertainty_analyzer: UncertaintyAnalyzer,
    output_path: str,
    no_report: bool,
    title: str,
    figures_dir: str = "./figures",
    no_figures: bool = False,
) -> None:
    """Finalise metrics, print summary, optionally write HTML report and figures."""
    print()
    summary   = evaluator.finalize()
    pearson_r = uncertainty_analyzer.compute_correlation()
    summary.pearson_r = pearson_r

    print(evaluator.generate_full_report())
    print()
    print(uncertainty_analyzer.generate_report())

    # ── Statistical analysis ──────────────────────────────────────────────
    try:
        from src.benchmark.stats import compute_all_stats, format_stats_table
        stats = compute_all_stats(summary)
        print()
        print(format_stats_table(stats))
    except Exception as e:
        logger.warning("Statistical analysis skipped: %s", e)

    # ── Figures ───────────────────────────────────────────────────────────
    if not no_figures:
        try:
            from src.reporting.visualize import generate_all_figures
            saved = generate_all_figures(
                summary,
                uncertainty_analyzer=uncertainty_analyzer,
                output_dir=figures_dir,
            )
            if saved:
                logger.info(
                    "Generated %d figures in %s/",
                    len(saved), os.path.abspath(figures_dir),
                )
        except Exception as e:
            logger.warning("Figure generation skipped: %s", e)

    # ── HTML report ───────────────────────────────────────────────────────
    if not no_report:
        report_gen = ReportGenerator(
            summary,
            uncertainty_analyzer=uncertainty_analyzer,
            pearson_r=pearson_r,
            title=title,
        )
        html = report_gen.generate()
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("HTML report saved → %s", os.path.abspath(output_path))


# ── Mode handlers ─────────────────────────────────────────────────────────────

def handle_demo(args: argparse.Namespace) -> None:
    """Synthetic data, 3 corruptions, ~30 s."""
    logger.info("Mode: DEMO  (synthetic data, no downloads)")
    device      = torch.device("cpu")
    mock_loader = MockCorruptionLoader(batch_size=args.batch_size, num_samples=640)
    model       = build_model(weights_path=args.weights_path, device=device)

    corruptions          = MOCK_CORRUPTION_TYPES
    evaluator            = BenchmarkEvaluator(corruptions, severity=args.severity)
    uncertainty_analyzer = UncertaintyAnalyzer(model, device)

    for corruption in corruptions:
        print()
        _run_single_corruption(
            model,
            lambda c, s: mock_loader.get_loader(c, s),
            corruption,
            args.severity,
            device,
            evaluator,
            uncertainty_analyzer,
            args.tent_lr,
            args.pl_threshold,
        )

    _finish(evaluator, uncertainty_analyzer, args.output, args.no_report,
            title="Domain Adaptation Benchmark — Demo",
            figures_dir=args.figures_dir, no_figures=args.no_figures)


def handle_full(args: argparse.Namespace) -> None:
    """Full benchmark: 15 corruptions × 4 methods on real CIFAR-10-C."""
    logger.info("Mode: FULL  (data-dir=%s, severity=%d)", args.data_dir, args.severity)

    if not os.path.isdir(args.data_dir):
        logger.error(
            "CIFAR-10-C directory not found: %s\n"
            "Download from https://zenodo.org/record/2535967 "
            "then extract the tar archive.", args.data_dir
        )
        sys.exit(1)

    device       = torch.device("cpu")
    real_loader  = CIFAR10CLoader(args.data_dir, batch_size=args.batch_size)
    available    = real_loader.available_corruptions()

    if not available:
        logger.error("No .npy corruption files found in %s", args.data_dir)
        sys.exit(1)

    logger.info("Found %d corruption types.", len(available))
    model                = build_model(weights_path=args.weights_path, device=device)
    evaluator            = BenchmarkEvaluator(available, severity=args.severity)
    uncertainty_analyzer = UncertaintyAnalyzer(model, device)

    t_total = time.perf_counter()
    for corruption in available:
        print()
        _run_single_corruption(
            model,
            real_loader.get_loader,
            corruption,
            args.severity,
            device,
            evaluator,
            uncertainty_analyzer,
            args.tent_lr,
            args.pl_threshold,
        )

    logger.info("Full benchmark completed in %.1f minutes.",
                (time.perf_counter() - t_total) / 60)
    _finish(evaluator, uncertainty_analyzer, args.output, args.no_report,
            title=f"Domain Adaptation Benchmark — Full (severity {args.severity})",
            figures_dir=args.figures_dir, no_figures=args.no_figures)


def handle_single_corruption(args: argparse.Namespace) -> None:
    """Evaluate one named corruption type."""
    logger.info(
        "Mode: SINGLE  (%s, severity %d)", args.corruption, args.severity
    )

    if not os.path.isdir(args.data_dir):
        logger.error("Data directory not found: %s", args.data_dir)
        sys.exit(1)

    device       = torch.device("cpu")
    real_loader  = CIFAR10CLoader(args.data_dir, batch_size=args.batch_size)
    model        = build_model(weights_path=args.weights_path, device=device)
    evaluator    = BenchmarkEvaluator([args.corruption], severity=args.severity)
    ua           = UncertaintyAnalyzer(model, device)

    _run_single_corruption(
        model,
        real_loader.get_loader,
        args.corruption,
        args.severity,
        device,
        evaluator,
        ua,
        args.tent_lr,
        args.pl_threshold,
    )

    _finish(evaluator, ua, args.output, args.no_report,
            title=f"{args.corruption} — severity {args.severity}",
            figures_dir=args.figures_dir, no_figures=args.no_figures)


def handle_dry_run(args: argparse.Namespace) -> None:
    """Check imports and model loading without running adaptation."""
    logger.info("Mode: DRY RUN")
    device = torch.device("cpu")

    logger.info("  Loading model …")
    model = build_model(weights_path=args.weights_path, device=device)
    logger.info("  ✓ Model loaded. Parameters: %s",
                f"{sum(p.numel() for p in model.model.parameters()):,}")

    logger.info("  Creating mock data loader …")
    loader = MockCorruptionLoader(batch_size=8, num_samples=16)
    dl     = loader.get_loader("gaussian_noise", severity=3)
    images, labels = next(iter(dl))
    logger.info("  ✓ Batch shape: %s  labels: %s", images.shape, labels.shape)

    logger.info("  Running single forward pass …")
    model.eval()
    with torch.no_grad():
        logits = model(images)
    logger.info("  ✓ Output shape: %s", logits.shape)

    logger.info("  Testing save/restore …")
    model.restore_original_state()
    logger.info("  ✓ Restore OK")

    logger.info("Dry run passed — all systems operational.")


def handle_fine_tune(args: argparse.Namespace) -> None:
    """Fine-tune ResNet-50 head on clean CIFAR-10."""
    logger.info(
        "Mode: FINE-TUNE  (epochs=%d, lr=%g, save=%s)",
        args.ft_epochs, args.ft_lr, args.weights_path,
    )
    device = torch.device("cpu")
    logger.info("Downloading / loading CIFAR-10 …")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir=args.cifar10_dir, batch_size=args.batch_size
    )
    model = build_model(device=device)

    fine_tune_on_cifar10(
        model,
        train_loader,
        device=device,
        epochs=args.ft_epochs,
        lr=args.ft_lr,
        save_path=args.weights_path,
    )

    from src.backbone.pretrained_model import evaluate_on_clean
    acc = evaluate_on_clean(model, test_loader, device)
    logger.info("Clean CIFAR-10 test accuracy: %.2f%%", 100 * acc)
    logger.info("Checkpoint saved to %s", args.weights_path)


# ── Full-sweep handler ────────────────────────────────────────────────────────

def handle_full_sweep(args: argparse.Namespace) -> None:
    """Full benchmark across ALL 5 severity levels (comprehensive research mode)."""
    logger.info("Mode: FULL-SWEEP  (all severities, data-dir=%s)", args.data_dir)

    if not os.path.isdir(args.data_dir):
        logger.error(
            "CIFAR-10-C directory not found: %s\n"
            "Download from https://zenodo.org/record/2535967 "
            "then extract the tar archive.", args.data_dir
        )
        sys.exit(1)

    device      = torch.device("cpu")
    real_loader = CIFAR10CLoader(args.data_dir, batch_size=args.batch_size)
    available   = real_loader.available_corruptions()

    if not available:
        logger.error("No .npy corruption files found in %s", args.data_dir)
        sys.exit(1)

    logger.info("Found %d corruption types. Running severities 1-5.", len(available))
    model = build_model(weights_path=args.weights_path, device=device)

    severity_summaries = {}
    t_total = time.perf_counter()

    for severity in range(1, 6):
        logger.info("\n" + "=" * 60)
        logger.info("  SEVERITY LEVEL %d / 5", severity)
        logger.info("=" * 60)

        evaluator = BenchmarkEvaluator(available, severity=severity)
        ua        = UncertaintyAnalyzer(model, device)

        for corruption in available:
            print()
            _run_single_corruption(
                model, real_loader.get_loader,
                corruption, severity, device,
                evaluator, ua,
                args.tent_lr, args.pl_threshold,
            )

        summary = evaluator.finalize()
        pearson_r = ua.compute_correlation()
        summary.pearson_r = pearson_r
        severity_summaries[severity] = summary

        logger.info(
            "Severity %d complete: mCE(baseline)=%.4f, mCE(best)=%.4f",
            severity, summary.baseline_mce, summary.best_mce,
        )

    total_mins = (time.perf_counter() - t_total) / 60
    logger.info("Full sweep completed in %.1f minutes.", total_mins)

    # Use severity 3 as the primary report
    primary = severity_summaries[3]
    primary_evaluator = BenchmarkEvaluator(available, severity=3)
    primary_ua = UncertaintyAnalyzer(model, device)

    # Rebuild evaluator for severity 3 for text report
    for corruption in available:
        for method in ["no_adaptation", "test_time_norm", "tent", "pseudo_label"]:
            acc = primary.accuracy_table.get(corruption, {}).get(method, 0.0)
            h = primary.entropy_table.get(corruption, {}).get(method, 0.0)
            primary_evaluator.add_result(corruption, method, acc, h)

    # Generate severity analysis figure
    try:
        from src.reporting.visualize import generate_all_figures
        os.makedirs(args.figures_dir, exist_ok=True)
        generate_all_figures(
            primary,
            severity_results=severity_summaries,
            output_dir=args.figures_dir,
        )
    except Exception as e:
        logger.warning("Figure generation skipped: %s", e)

    # Print severity comparison table
    print("\n" + "=" * 70)
    print("  SEVERITY SWEEP SUMMARY")
    print("=" * 70)
    print(f"{'Severity':<10} {'No Adapt mCE':<15} {'Best mCE':<12} "
          f"{'Best Method':<15} {'Pearson r':<12}")
    print("-" * 70)
    for sev in range(1, 6):
        s = severity_summaries[sev]
        bm_name = METHOD_DISPLAY.get(s.best_method, s.best_method)
        print(f"{sev:<10} {s.baseline_mce:<15.4f} {s.best_mce:<12.4f} "
              f"{bm_name:<15} {s.pearson_r:<+12.4f}")
    print()

    # Write HTML report for severity 3
    if not args.no_report:
        report_gen = ReportGenerator(
            primary,
            pearson_r=primary.pearson_r,
            title="Domain Adaptation Benchmark — Full Sweep (severity 1-5)",
        )
        html = report_gen.generate()
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(html)
        logger.info("HTML report saved → %s", os.path.abspath(args.output))


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Default to demo if nothing specified
    if not any([args.mode, args.corruption, args.dry_run, args.fine_tune]):
        logger.info("No mode specified — defaulting to --mode demo")
        args.mode = "demo"

    if args.dry_run:
        handle_dry_run(args)
    elif args.fine_tune:
        handle_fine_tune(args)
    elif args.corruption:
        handle_single_corruption(args)
    elif args.mode == "demo":
        handle_demo(args)
    elif args.mode == "full":
        handle_full(args)
    elif args.mode == "full-sweep":
        handle_full_sweep(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
