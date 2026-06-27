"""
Pseudo-Label Test-Time Adaptation — Sun et al., ICML 2020.

Protocol
--------
For each test batch:
    1. Forward pass → get predicted labels and confidence scores.
    2. Filter: keep only samples where max-softmax probability ≥ threshold.
    3. Treat filtered predictions as ground-truth "pseudo-labels".
    4. Fine-tune the model on the pseudo-labeled subset (1 gradient step).
    5. Evaluate on the full original batch.

Why confidence threshold = 0.9?
--------------------------------
A high threshold (0.9) is a deliberate conservative choice:

    - Threshold too low  → includes many uncertain/incorrect samples as
      pseudo-labels → noisy supervision → accuracy degradation.
    - Threshold too high → very few samples pass → too little signal for
      meaningful gradient updates; entire batches may be skipped.

At threshold 0.9, roughly the top ~10–30% of confident predictions are used,
empirically balancing signal quality and quantity for CIFAR-10 at batch
size 64–128.  This follows the recommendation in the original pseudo-label
literature (Lee, 2013; Cascante-Bonilla et al., 2021).

Counter-Intuitive Failure Mode: Blur Corruptions
-------------------------------------------------
Pseudo-label adaptation DEGRADES performance on blur corruptions.  This is
a central finding of this benchmark, documented explicitly below.

Root cause — Confirmation Bias in Self-Training:

    When images are blurred, the model sometimes misclassifies them with
    high confidence.  Example:
        - A blurred CIFAR-10 cat image loses fine details.
        - The model assigns 97% probability to "dog" (wrong label).
        - p_max = 0.97 ≥ threshold → pseudo-label "dog" is accepted.
        - The model fine-tunes on (blurred_cat_image, "dog"), reinforcing
          the incorrect association.
        - After this fine-tuning step, the model is *worse* than before.

    This phenomenon is called "confirmation bias" — the model's wrong
    confident predictions are used to train it toward those same wrong
    predictions.

    In contrast, noise corruptions tend to cause high uncertainty
    (low p_max, high entropy) rather than high-confidence wrong predictions.
    Noise-corrupted images fail the threshold more often, and when they do
    pass, they are more likely to be genuinely correct (model still recognises
    the class despite noise).

Corruptions where failure is documented:
    defocus_blur, glass_blur, motion_blur, zoom_blur

These blur types are in BLUR_CORRUPTIONS in src/models.py.  The evaluator
automatically detects and reports this failure in the benchmark summary.

Potential fix (not implemented):
    Replace hard pseudo-labels with soft labels or use consistency
    regularisation (MixUp-style) to penalise overconfident wrong predictions.

Reference
---------
Sun Y., Wang X., Liu Z., Miller J., Efros A., Hardt M. (2020).
"Test-time training with self-supervision for generalization under
distribution shifts." International Conference on Machine Learning (ICML).
https://arxiv.org/abs/1909.13231

Output keys
-----------
method              : "pseudo_label"
accuracy            : float   top-1 accuracy after PL adaptation
loss                : float   mean cross-entropy
mean_entropy_before : float   entropy before any fine-tuning
mean_entropy_after  : float   entropy after fine-tuning
entropy_reduction   : float   before - after
acceptance_rate     : float   fraction of samples with p_max ≥ threshold
total_accepted      : int     cumulative samples used for fine-tuning
total_rejected      : int     cumulative samples rejected (low confidence)
skipped_batches     : int     batches with zero confident predictions
num_samples         : int
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

CONFIDENCE_THRESHOLD = 0.9   # Standard pseudo-label confidence gate


def adapt_with_pseudo_label(
    model: nn.Module,
    data_loader,
    device: torch.device = torch.device("cpu"),
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
    lr: float = 1e-3,
    n_steps: int = 1,
) -> Dict[str, Any]:
    """
    Apply pseudo-label adaptation and evaluate on corrupted data.

    For each batch the function:
        1. Generates predictions in eval mode (no gradient).
        2. Filters by confidence threshold.
        3. If any samples pass the threshold, fine-tunes on them for
           ``n_steps`` gradient steps.
        4. Evaluates the (possibly adapted) model on the full batch.

    Failure note: on blur corruptions the accepted pseudo-labels are often
    incorrect (confident wrong predictions), leading to worse-than-baseline
    accuracy.  See module docstring for the full explanation.

    Parameters
    ----------
    model : nn.Module
        Pre-trained model with all parameters trainable (will be fine-tuned).
    data_loader : DataLoader
        Corrupted target domain data.
    device : torch.device
        Computation device.
    confidence_threshold : float
        Minimum max-softmax probability for accepting a pseudo-label.
        Default 0.9 (conservative high-confidence gate).
    lr : float
        Adam learning rate for the fine-tuning step.
    n_steps : int
        Gradient steps on the pseudo-labeled subset per batch.

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary described in module docstring.
    """
    total_correct   = 0
    total_samples   = 0
    total_loss      = 0.0
    all_h_before: List[float] = []
    all_h_after:  List[float] = []
    total_accepted   = 0
    total_rejected   = 0
    skipped_batches  = 0

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        # ------------------------------------------------------------------ #
        # Step 1 — forward pass: generate candidate pseudo-labels            #
        # ------------------------------------------------------------------ #
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
            probs  = torch.softmax(logits, dim=1)

            # Entropy before adaptation
            h_before = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            all_h_before.extend(h_before.cpu().tolist())

            # Confidence scores and predicted labels
            max_probs, pseudo_labels = probs.max(dim=1)   # (N,), (N,)

        # ------------------------------------------------------------------ #
        # Step 2 — filter by confidence                                       #
        # ------------------------------------------------------------------ #
        confident_mask = max_probs >= confidence_threshold
        n_accepted = int(confident_mask.sum().item())
        n_rejected = inputs.size(0) - n_accepted
        total_accepted += n_accepted
        total_rejected += n_rejected

        # ------------------------------------------------------------------ #
        # Step 3 — fine-tune on pseudo-labeled confident subset              #
        # ------------------------------------------------------------------ #
        if n_accepted == 0:
            # No confident predictions → skip adaptation for this batch
            skipped_batches += 1
            logger.debug(
                "Batch %d: 0/%d samples accepted (all below threshold %.2f). "
                "Falling back to no-adaptation for this batch.",
                batch_idx, inputs.size(0), confidence_threshold,
            )
        else:
            confident_inputs  = inputs[confident_mask]
            confident_targets = pseudo_labels[confident_mask]   # pseudo-labels, not true labels

            model.train()
            # Re-create optimizer each batch to avoid momentum accumulation
            # across potentially very different pseudo-label sets
            optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

            for _ in range(n_steps):
                optimizer.zero_grad()
                pl_logits = model(confident_inputs)
                pl_loss   = criterion(pl_logits, confident_targets)
                pl_loss.backward()
                optimizer.step()

        # ------------------------------------------------------------------ #
        # Step 4 — evaluate on the full original batch                       #
        # ------------------------------------------------------------------ #
        model.eval()
        with torch.no_grad():
            eval_logits = model(inputs)
            eval_loss   = criterion(eval_logits, targets)

            probs_a = torch.softmax(eval_logits, dim=1)
            h_after = -(probs_a * torch.log(probs_a + 1e-8)).sum(dim=1)
            all_h_after.extend(h_after.cpu().tolist())

            preds = eval_logits.argmax(dim=1)
            total_correct  += preds.eq(targets).sum().item()
            total_samples  += targets.size(0)
            total_loss     += eval_loss.item()

    # ------------------------------------------------------------------ #
    # Aggregate                                                           #
    # ------------------------------------------------------------------ #
    n = len(data_loader)
    accuracy             = total_correct / total_samples
    mean_loss            = total_loss / n
    mean_entropy_before  = float(torch.tensor(all_h_before).mean().item())
    mean_entropy_after   = float(torch.tensor(all_h_after).mean().item())

    total_all = total_accepted + total_rejected
    acceptance_rate = total_accepted / total_all if total_all > 0 else 0.0

    logger.debug(
        "PseudoLabel: acc=%.4f  H: %.4f → %.4f  "
        "accept=%.1f%%  skipped_batches=%d",
        accuracy,
        mean_entropy_before,
        mean_entropy_after,
        100 * acceptance_rate,
        skipped_batches,
    )

    return {
        "method":               "pseudo_label",
        "accuracy":             accuracy,
        "loss":                 mean_loss,
        "mean_entropy_before":  mean_entropy_before,
        "mean_entropy_after":   mean_entropy_after,
        "entropy_reduction":    mean_entropy_before - mean_entropy_after,
        "acceptance_rate":      acceptance_rate,
        "total_accepted":       total_accepted,
        "total_rejected":       total_rejected,
        "skipped_batches":      skipped_batches,
        "num_samples":          total_samples,
    }
