"""
TENT: Fully Test-Time Adaptation by Entropy Minimisation — Wang et al., ICLR 2021.

Why Entropy Minimisation Works for Domain Adaptation
-----------------------------------------------------
A model trained on clean data (source domain) learns to assign high probability
mass to the correct class for clean images.  Prediction entropy is low:
    H(p) = -Σ p_i log p_i ≈ 0   (model is confident and correct)

Under distribution shift (corrupted target domain), the model becomes
"confused" — probability mass spreads across classes, raising entropy:
    H(p) = -Σ p_i log p_i → log(C)   (approaching uniform, maximum uncertainty)

TENT exploits this signal without requiring any ground-truth labels:

    Loss  =  H(p)  =  -Σ softmax(logits)_i · log_softmax(logits)_i

Minimising this loss forces the model to commit to confident predictions.
Because the gradient flows through the batch normalisation affine parameters
(γ, β), TENT effectively recalibrates the scale and shift of every feature
map toward the test distribution.  This is entirely self-supervised — the
only signal comes from the test data itself.

Why only update BN affine parameters?
--------------------------------------
If all weights were updated, the model would rapidly overfit to individual
batches (catastrophic forgetting).  BN affine parameters (γ, β) are:
    1. Lightweight (2 × C scalars per layer, vs millions of conv weights)
    2. Distribution-sensitive (they directly modulate feature statistics)
    3. Easily resettable (we snapshot and restore between corruptions)

TENT adaptation loop (per batch):
    1. Freeze all parameters
    2. Unfreeze BN γ, β only
    3. Set BN layers to training mode (stat update + affine gradient)
    4. Forward pass → compute entropy loss H(p)
    5. Backward pass + Adam step → update γ, β
    6. Evaluate predictions (eval mode, updated BN stats + affine params)

Model reset between corruptions:
    After each corruption type, the model is restored to its original state
    (pre-adaptation weights) to prevent cross-corruption contamination —
    the adaptation gain for corruption B must not be inflated by adaptation
    already done for corruption A.

Reference
---------
Wang D., Shelhamer E., Liu S., Olshausen B., Darrell T. (2021).
"Tent: Fully test-time adaptation by entropy minimization."
International Conference on Learning Representations (ICLR).
https://arxiv.org/abs/2006.10726

Output keys
-----------
method              : "tent"
accuracy            : float   top-1 accuracy after TENT
loss                : float   mean cross-entropy (evaluated after adaptation)
mean_entropy_before : float   mean H(p) before any TENT step
mean_entropy_after  : float   mean H(p) after TENT step (should decrease)
entropy_reduction   : float   before - after  (positive = TENT adapted)
mean_gradient_norm  : float   mean L2 norm of BN param gradients
num_samples         : int
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _configure_tent(model: nn.Module, lr: float) -> optim.Adam:
    """
    Prepare model for TENT:
        1. Freeze all parameters.
        2. Unfreeze BN affine params (γ, β) and set BN layers to train mode.
        3. Return Adam optimizer for BN affine params only.

    Parameters
    ----------
    model : nn.Module
        Model to configure in-place.
    lr : float
        Adam learning rate.

    Returns
    -------
    optim.Adam
        Optimizer controlling only BN affine parameters.
    """
    # Step 1: Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    # Step 2: BN layers → train mode, enable affine gradients
    trainable: List[torch.Tensor] = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()
            if m.weight is not None:
                m.weight.requires_grad_(True)
                trainable.append(m.weight)
            if m.bias is not None:
                m.bias.requires_grad_(True)
                trainable.append(m.bias)

    # Step 3: Build optimizer
    return optim.Adam(trainable, lr=lr, betas=(0.9, 0.999))


def tent_entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    """
    TENT loss: mean per-sample Shannon entropy.

    Uses numerically stable log_softmax instead of log(softmax(·)):
        H(p) = -Σ softmax(x)_i · log_softmax(x)_i

    Parameters
    ----------
    logits : torch.Tensor
        Shape (N, C) — raw pre-softmax model outputs.

    Returns
    -------
    torch.Tensor
        Scalar entropy loss (mean over batch).
    """
    probs     = torch.softmax(logits, dim=1)
    log_probs = torch.log_softmax(logits, dim=1)
    return -(probs * log_probs).sum(dim=1).mean()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def adapt_with_tent(
    model: nn.Module,
    data_loader,
    device: torch.device = torch.device("cpu"),
    lr: float = 1e-3,
    n_steps: int = 1,
) -> Dict[str, Any]:
    """
    Apply TENT entropy minimisation and evaluate on corrupted data.

    For each batch:
        1. Record pre-adaptation entropy (no-grad, eval mode).
        2. Configure model: freeze all, unfreeze BN γ/β, set BN to train.
        3. ``n_steps`` forward-backward passes to minimise H(p).
        4. Switch to eval mode, evaluate predictions (post-adaptation).

    The model is NOT reset between batches within a single corruption run —
    BN affine parameters accumulate adaptation across batches.  The caller
    must call ``model.restore_original_state()`` before the next
    (corruption, method) pair to ensure independent evaluation.

    Parameters
    ----------
    model : nn.Module
        Pre-trained model with BatchNorm2d layers.
    data_loader : DataLoader
        Corrupted target domain data.
    device : torch.device
        Computation device.
    lr : float
        Adam learning rate for BN affine parameter updates.
        Default 1e-3 follows the original TENT paper.
    n_steps : int
        Gradient steps per batch (default 1 as in TENT paper).
        More steps increase adaptation but risk overfitting to the batch.

    Returns
    -------
    Dict[str, Any]
        Metrics dictionary described in module docstring.
    """
    optimizer = _configure_tent(model, lr=lr)

    total_correct  = 0
    total_samples  = 0
    total_loss     = 0.0
    all_entropies_before: List[float] = []
    all_entropies_after:  List[float] = []
    gradient_norms: List[float] = []

    criterion = nn.CrossEntropyLoss()

    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        # ------------------------------------------------------------------ #
        # Step 1 — pre-adaptation entropy (eval mode, no grad)               #
        # ------------------------------------------------------------------ #
        model.eval()
        # Temporarily freeze BN in eval mode for clean entropy measurement
        with torch.no_grad():
            logits_before = model(inputs)
            probs_b       = torch.softmax(logits_before, dim=1)
            h_before      = -(probs_b * torch.log(probs_b + 1e-8)).sum(dim=1)
            all_entropies_before.extend(h_before.cpu().tolist())

        # ------------------------------------------------------------------ #
        # Step 2 — restore BN to training mode after eval() call             #
        # ------------------------------------------------------------------ #
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.train()

        # ------------------------------------------------------------------ #
        # Step 3 — TENT gradient step(s): minimise entropy                   #
        # ------------------------------------------------------------------ #
        for _ in range(n_steps):
            optimizer.zero_grad()
            logits = model(inputs)
            loss   = tent_entropy_loss(logits)
            loss.backward()

            # Compute gradient norm for diagnostics
            total_sq_norm = sum(
                p.grad.data.norm(2).item() ** 2
                for p in model.parameters()
                if p.grad is not None
            )
            gradient_norms.append(total_sq_norm ** 0.5)

            optimizer.step()

        # ------------------------------------------------------------------ #
        # Step 4 — evaluate with updated BN params                           #
        # ------------------------------------------------------------------ #
        model.eval()
        with torch.no_grad():
            logits_eval = model(inputs)
            eval_loss   = criterion(logits_eval, targets)

            probs_a = torch.softmax(logits_eval, dim=1)
            h_after = -(probs_a * torch.log(probs_a + 1e-8)).sum(dim=1)
            all_entropies_after.extend(h_after.cpu().tolist())

            preds = logits_eval.argmax(dim=1)
            total_correct  += preds.eq(targets).sum().item()
            total_samples  += targets.size(0)
            total_loss     += eval_loss.item()

    # ------------------------------------------------------------------ #
    # Aggregate                                                           #
    # ------------------------------------------------------------------ #
    n = len(data_loader)
    accuracy             = total_correct / total_samples
    mean_loss            = total_loss / n
    mean_entropy_before  = float(torch.tensor(all_entropies_before).mean().item())
    mean_entropy_after   = float(torch.tensor(all_entropies_after).mean().item())
    mean_grad_norm       = float(
        torch.tensor(gradient_norms).mean().item()
    ) if gradient_norms else 0.0

    logger.debug(
        "TENT: acc=%.4f  H: %.4f → %.4f (Δ=%.4f)  grad_norm=%.6f",
        accuracy,
        mean_entropy_before,
        mean_entropy_after,
        mean_entropy_before - mean_entropy_after,
        mean_grad_norm,
    )

    return {
        "method":               "tent",
        "accuracy":             accuracy,
        "loss":                 mean_loss,
        "mean_entropy_before":  mean_entropy_before,
        "mean_entropy_after":   mean_entropy_after,
        "entropy_reduction":    mean_entropy_before - mean_entropy_after,
        "mean_gradient_norm":   mean_grad_norm,
        "num_samples":          total_samples,
    }
