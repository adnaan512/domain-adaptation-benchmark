"""
Baseline: Direct inference on corrupted data with no adaptation.

This establishes the performance floor that all TTA methods must exceed
to justify their computational overhead.  Every accuracy reported for
TTN, TENT, and pseudo-label is compared against this baseline to compute
the "adaptation gain".

The baseline is intentionally simple: the model trained on clean CIFAR-10
is applied as-is to corrupted images.  Its batch normalisation layers use
the running statistics estimated on the clean training set, which are
mismatched to the corrupted test distribution — this is the root cause of
the accuracy drop that TTA methods aim to correct.

Output keys
-----------
method          : "no_adaptation"
accuracy        : float    top-1 accuracy in [0, 1]
loss            : float    mean cross-entropy
mean_entropy    : float    mean Shannon entropy before adaptation (≡ at inference)
std_entropy     : float    standard deviation of per-sample entropy
mean_max_prob   : float    mean of max-softmax probability
num_samples     : int      number of evaluation samples
all_entropies   : list     per-sample entropies (for histogram, correlation)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def evaluate_no_adaptation(
    model: nn.Module,
    data_loader,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Evaluate a model on corrupted data without any adaptation.

    Parameters
    ----------
    model : nn.Module
        Pre-trained model.  Must already be in eval mode.
        Model state is not modified by this function.
    data_loader : DataLoader
        DataLoader yielding (images, labels) from the corrupted domain.
    device : torch.device
        Computation device (cpu or cuda).

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary described in module docstring.
    """
    model.eval()

    total_correct  = 0
    total_samples  = 0
    total_loss     = 0.0
    all_entropies: List[float] = []
    all_max_probs: List[float] = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs  = inputs.to(device)
            targets = targets.to(device)

            logits = model(inputs)

            # Cross-entropy loss
            loss = criterion(logits, targets)
            total_loss += loss.item()

            # Predictions
            preds = logits.argmax(dim=1)
            total_correct += preds.eq(targets).sum().item()
            total_samples += targets.size(0)

            # Entropy and confidence
            probs     = torch.softmax(logits, dim=1)
            log_probs = torch.log(probs + 1e-8)
            entropy   = -(probs * log_probs).sum(dim=1)           # (N,)
            max_probs = probs.max(dim=1).values                   # (N,)

            all_entropies.extend(entropy.cpu().tolist())
            all_max_probs.extend(max_probs.cpu().tolist())

    n = len(data_loader)
    entropy_t  = torch.tensor(all_entropies)
    max_prob_t = torch.tensor(all_max_probs)

    accuracy     = total_correct / total_samples
    mean_entropy = float(entropy_t.mean().item())
    std_entropy  = float(entropy_t.std().item())
    mean_max_prob = float(max_prob_t.mean().item())

    logger.debug(
        "no_adaptation: acc=%.4f  H=%.4f±%.4f  max_p=%.4f",
        accuracy, mean_entropy, std_entropy, mean_max_prob,
    )

    return {
        "method":        "no_adaptation",
        "accuracy":      accuracy,
        "loss":          total_loss / n,
        "mean_entropy":  mean_entropy,
        "std_entropy":   std_entropy,
        "mean_max_prob": mean_max_prob,
        "num_samples":   total_samples,
        "all_entropies": all_entropies,
        # Aliases for uniform interface with adaptation methods
        "mean_entropy_before": mean_entropy,
        "mean_entropy_after":  mean_entropy,
        "entropy_reduction":   0.0,
    }
