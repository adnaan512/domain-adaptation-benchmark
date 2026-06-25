"""
Test-Time Normalization (TTN) — Schneider et al., NeurIPS 2020.

Core Idea
---------
Batch normalisation layers store running statistics (mean μ, variance σ²)
estimated on the source (training) domain.  When a model encounters corrupted
images whose pixel distribution differs from the training set, these cached
statistics are mismatched, leading to distorted activations and degraded
predictions.

TTN corrects this mismatch without labels or gradients:

    for each test batch:
        model.train()            # BN computes batch statistics (train mode)
        with torch.no_grad():    # no gradient tape
            _ = model(x)         # forward pass updates running_mean / running_var
        model.eval()             # switch back; now BN uses updated statistics
        predictions = model(x)  # second forward pass with corrected BN stats

The two forward passes are the defining cost of TTN: one to adapt statistics,
one to obtain final predictions.  No optimizer is required.

Why does this work?
The training-mode forward pass triggers BN's exponential moving average update:

    running_mean ← (1 - momentum) · running_mean + momentum · batch_mean
    running_var  ← (1 - momentum) · running_var  + momentum · batch_var

After one batch the running statistics are partially aligned to the test
distribution, reducing the normalisation mismatch.

Limitations vs TENT:
    - Adapts only running statistics, not affine parameters (γ, β).
    - Requires two forward passes per batch (≈2× inference cost of baseline).
    - Single-batch adaptation leaves statistics partially mismatched;
      a large test batch (≥128 samples) is needed for reliable estimates.

Reference
---------
Schneider et al. (2020). "Improving robustness against common corruptions
by covariate shift adaptation." Advances in Neural Information Processing
Systems (NeurIPS), 33, 11539–11551.

Output keys
-----------
method              : "test_time_norm"
accuracy            : float   top-1 accuracy after TTN
loss                : float   mean cross-entropy
mean_entropy_before : float   entropy before BN stat update
mean_entropy_after  : float   entropy after BN stat update
entropy_reduction   : float   before - after  (positive = adapted)
num_samples         : int
bn_update_count     : int     number of BN layer updates across all batches
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def adapt_with_ttn(
    model: nn.Module,
    data_loader,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Apply Test-Time Normalisation and evaluate on corrupted data.

    The model is returned to eval mode after every batch.  Running statistics
    accumulate across batches during the evaluation loop (exponential moving
    average), giving progressively better alignment to the test distribution.

    The model's affine parameters (γ, β) and all other weights are unchanged.
    The caller is responsible for restoring original BN stats via
    ``model.restore_original_state()`` before the next corruption/method run.

    Parameters
    ----------
    model : nn.Module
        Pre-trained neural network.  Must have BatchNorm2d layers.
    data_loader : DataLoader
        Corrupted target domain data loader.
    device : torch.device
        Computation device.

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary described in module docstring.
    """
    # Count BN layers for diagnostics
    bn_layers     = [m for m in model.modules() if isinstance(m, nn.BatchNorm2d)]
    num_bn_layers = len(bn_layers)

    total_correct  = 0
    total_samples  = 0
    total_loss     = 0.0
    all_entropies_before: List[float] = []
    all_entropies_after:  List[float] = []
    bn_update_count = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        # ------------------------------------------------------------------ #
        # Step 1 — record pre-adaptation entropy                             #
        # ------------------------------------------------------------------ #
        model.eval()
        with torch.no_grad():
            logits_before = model(inputs)
            probs_b       = torch.softmax(logits_before, dim=1)
            h_before      = -(probs_b * torch.log(probs_b + 1e-8)).sum(dim=1)
            all_entropies_before.extend(h_before.cpu().tolist())

        # ------------------------------------------------------------------ #
        # Step 2 — TTN forward pass (update BN running statistics)           #
        # ------------------------------------------------------------------ #
        # model.train() activates BN's exponential-average update rule.
        # torch.no_grad() prevents gradient accumulation — only running_mean
        # and running_var are updated, not any learnable weights.
        model.train()
        with torch.no_grad():
            _ = model(inputs)   # sole purpose: trigger BN stat update
        bn_update_count += num_bn_layers

        # ------------------------------------------------------------------ #
        # Step 3 — evaluate with updated BN statistics                       #
        # ------------------------------------------------------------------ #
        model.eval()
        with torch.no_grad():
            logits_eval = model(inputs)
            loss        = criterion(logits_eval, targets)

            probs_a = torch.softmax(logits_eval, dim=1)
            h_after = -(probs_a * torch.log(probs_a + 1e-8)).sum(dim=1)
            all_entropies_after.extend(h_after.cpu().tolist())

            preds = logits_eval.argmax(dim=1)
            total_correct  += preds.eq(targets).sum().item()
            total_samples  += targets.size(0)
            total_loss     += loss.item()

    # ------------------------------------------------------------------ #
    # Aggregate                                                           #
    # ------------------------------------------------------------------ #
    n = len(data_loader)
    accuracy             = total_correct / total_samples
    mean_loss            = total_loss / n
    mean_entropy_before  = float(torch.tensor(all_entropies_before).mean().item())
    mean_entropy_after   = float(torch.tensor(all_entropies_after).mean().item())

    logger.debug(
        "TTN: acc=%.4f  H: %.4f → %.4f  BN_updates=%d",
        accuracy, mean_entropy_before, mean_entropy_after, bn_update_count,
    )

    return {
        "method":              "test_time_norm",
        "accuracy":            accuracy,
        "loss":                mean_loss,
        "mean_entropy_before": mean_entropy_before,
        "mean_entropy_after":  mean_entropy_after,
        "entropy_reduction":   mean_entropy_before - mean_entropy_after,
        "num_samples":         total_samples,
        "bn_update_count":     bn_update_count,
    }
