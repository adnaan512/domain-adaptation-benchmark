"""
Domain Adaptation Benchmark — Demo Mode
========================================
Runs the full 4-method pipeline on 3 synthetic corruptions.
No file downloads required.  Completes in ~30 seconds on CPU.

Usage
-----
    python examples/run_demo.py
    python examples/run_demo.py --output my_report.html
    python examples/run_demo.py --batch-size 32 --num-samples 320

Output
------
    demo_report.html  (self-contained dark HTML report)
    Console text summary

Notes
-----
- Results are not scientifically meaningful (random labels, untrained head).
- The adaptation pipeline is identical to the full --mode full benchmark.
- Use this to verify the installation and explore the report format.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Allow running as: python examples/run_demo.py  (from project root)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.backbone.pretrained_model import build_model
from src.data.dataset_loader import MockCorruptionLoader
from src.adaptation.no_adaptation import evaluate_no_adaptation
from src.adaptation.test_time_norm import adapt_with_ttn
from src.adaptation.tent import adapt_with_tent
from src.adaptation.pseudo_label import adapt_with_pseudo_label
from src.benchmark.evaluator import BenchmarkEvaluator
from src.uncertainty.uncertainty_analyzer import UncertaintyAnalyzer
from src.reporting.report_generator import ReportGenerator

# ── Config ────────────────────────────────────────────────────────────────────

DEMO_CORRUPTIONS = ["gaussian_noise", "blur", "brightness"]
DEMO_SEVERITY    = 3

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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Domain Adaptation Benchmark — Demo (no downloads)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--output", default="demo_report.html",
        help="Path for the HTML report output file.",
    )
    p.add_argument(
        "--batch-size", type=int, default=64,
        help="Mini-batch size for the mock data loaders.",
    )
    p.add_argument(
        "--num-samples", type=int, default=640,
        help="Number of synthetic samples per corruption loader.",
    )
    p.add_argument(
        "--weights-path", default=None,
        help="Optional path to a fine-tuned CIFAR-10 checkpoint (.pth). "
             "If not provided, uses ImageNet backbone with a random head.",
    )
    p.add_argument(
        "--tent-lr", type=float, default=1e-3,
        help="Adam learning rate for TENT adaptation.",
    )
    p.add_argument(
        "--pl-threshold", type=float, default=0.9,
        help="Confidence threshold for pseudo-label acceptance.",
    )
    return p.parse_args()


# ── Main ─────────────────────────────────────────────────────────────────────


def run_demo(args: argparse.Namespace) -> None:
    # Ensure UTF-8 printing for Windows console
    if sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except AttributeError:
            pass

    t_start = time.perf_counter()
    device  = torch.device("cpu")

    _banner()

    # ── 1. Build model ────────────────────────────────────────────────────
    logger.info("Loading ResNet-50 backbone …")
    model = build_model(weights_path=args.weights_path, device=device)
    n_params = sum(p.numel() for p in model.model.parameters())
    logger.info("  Parameters: %s", f"{n_params:,}")

    # ── 2. Prepare loaders ────────────────────────────────────────────────
    mock_loader = MockCorruptionLoader(
        batch_size=args.batch_size,
        num_samples=args.num_samples,
    )

    # ── 3. Run benchmark ─────────────────────────────────────────────────
    evaluator            = BenchmarkEvaluator(DEMO_CORRUPTIONS, severity=DEMO_SEVERITY)
    uncertainty_analyzer = UncertaintyAnalyzer(model, device)
    results_store: dict  = {}

    for corruption in DEMO_CORRUPTIONS:
        print()
        logger.info("━━━  Corruption: %s  (severity %d)  ━━━", corruption, DEMO_SEVERITY)
        results_store[corruption] = {}

        for method_name, method_fn in METHODS.items():
            # Reset model to original state before each method
            model.restore_original_state()
            model.eval()

            loader = mock_loader.get_loader(corruption, DEMO_SEVERITY)

            t0 = time.perf_counter()

            # Pass extra kwargs only to methods that accept them
            if method_name == "tent":
                result = method_fn(model, loader, device, lr=args.tent_lr)
            elif method_name == "pseudo_label":
                result = method_fn(model, loader, device,
                                   confidence_threshold=args.pl_threshold)
            else:
                result = method_fn(model, loader, device)

            elapsed = time.perf_counter() - t0

            acc            = result["accuracy"]
            entropy_before = result.get("mean_entropy",
                             result.get("mean_entropy_before", 0.0))

            evaluator.add_result(corruption, method_name, acc, entropy_before)
            results_store[corruption][method_name] = result

            logger.info(
                "  %-20s  acc=%6.2f%%  H_before=%.4f  [%.1fs]",
                METHOD_DISPLAY.get(method_name, method_name),
                100 * acc,
                entropy_before,
                elapsed,
            )

        # ── Uncertainty analysis (uses no-adaptation run) ─────────────────
        model.restore_original_state()
        model.eval()
        ua_loader       = mock_loader.get_loader(corruption, DEMO_SEVERITY)
        entropy_metrics = uncertainty_analyzer.compute_entropy(ua_loader)

        baseline_acc = results_store[corruption]["no_adaptation"]["accuracy"]
        tent_acc     = results_store[corruption]["tent"]["accuracy"]
        uncertainty_analyzer.record(
            corruption, entropy_metrics, baseline_acc, tent_acc
        )

    # ── 4. Finalise metrics ───────────────────────────────────────────────
    print()
    logger.info("Finalising benchmark metrics …")
    summary    = evaluator.finalize()
    pearson_r  = uncertainty_analyzer.compute_correlation()
    summary.pearson_r = pearson_r

    # ── 5. Print console report ───────────────────────────────────────────
    print()
    print(evaluator.generate_full_report())
    print()
    print(uncertainty_analyzer.generate_report())

    # ── 6. Generate HTML report ───────────────────────────────────────────
    print()
    logger.info("Generating HTML report …")
    report_gen = ReportGenerator(
        summary,
        uncertainty_analyzer=uncertainty_analyzer,
        pearson_r=pearson_r,
        title="Domain Adaptation Benchmark — Demo",
    )
    html_content = report_gen.generate()

    output_path = args.output
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html_content)

    t_total = time.perf_counter() - t_start

    print()
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  Demo complete in %.1f seconds", t_total)
    logger.info("  HTML report  → %s", os.path.abspath(output_path))
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print()
    print("  Open the report in your browser:")
    print(f"    file://{os.path.abspath(output_path)}")
    print()


def _banner() -> None:
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   Domain Adaptation Benchmark — Demo Mode            ║")
    print("║   Test-Time Adaptation for Distribution Shift        ║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║   Methods:   No Adapt · TTN · TENT · Pseudo-Label    ║")
    print("║   Data:      Synthetic (no download needed)           ║")
    print("║   Device:    CPU                                      ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


if __name__ == "__main__":
    run_demo(parse_args())
