"""
ResNet-50 backbone for the Domain Adaptation Benchmark.

Architecture
------------
Standard ResNet-50 with three CIFAR-10-specific modifications:
    1. First conv: 3×3 kernel, stride 1, padding 1  (ImageNet uses 7×7/2)
    2. MaxPool removed (32×32 inputs are too small for 3×3 max-pool)
    3. FC head: 2048 → 10  (CIFAR-10 classes)

Batch normalisation layers are preserved throughout — they are the primary
adaptation targets for TTN and TENT, so their structure must not be altered.

Model lifecycle
---------------
    model = build_model(weights_path="./cifar10_resnet50.pth")
    # model.save_original_state() is called inside build_model()

    # Before each adaptation method:
    model.restore_original_state()
    model.eval()

    # After TENT/pseudo-label adaptation:
    # model weights have changed; restore before next method.

Fine-tuning
-----------
When no saved checkpoint is found, the model can be fine-tuned on CIFAR-10
via fine_tune_on_cifar10(). Only the final FC layer and layer4 (+ BN layers)
are updated to preserve ImageNet-learned features.

    python -c "
    from src.backbone.pretrained_model import fine_tune_and_save
    fine_tune_and_save('./data', save_path='./cifar10_resnet50.pth')
    "
"""

from __future__ import annotations

import copy
import logging
import os
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models

logger = logging.getLogger(__name__)

NUM_CLASSES = 10


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class CIFAR10ResNet(nn.Module):
    """
    ResNet-50 adapted for CIFAR-10 (32 × 32 input).

    Exposes helpers used by TTN, TENT, and pseudo-label adapters:
        - get_batch_norm_layers()      → list of all BN layers
        - get_entropy(logits)          → per-sample entropy tensor
        - freeze_all_params()          → disable all gradients
        - unfreeze_bn_affine_params()  → re-enable BN γ, β only
        - save_original_state()        → snapshot current weights
        - restore_original_state()     → revert to snapshot
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        # ---- Build backbone ------------------------------------------------
        if pretrained:
            try:
                backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
                logger.info("Loaded ImageNet pretrained ResNet-50 weights.")
            except Exception as e:
                logger.warning(
                    "Could not download ImageNet weights (%s). "
                    "Falling back to random initialisation. "
                    "Set pretrained=False to suppress this warning.", e
                )
                backbone = models.resnet50(weights=None)
        else:
            backbone = models.resnet50(weights=None)

        # CIFAR-10 adaptations
        backbone.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        backbone.maxpool = nn.Identity()  # type: ignore[assignment]
        backbone.fc      = nn.Linear(backbone.fc.in_features, NUM_CLASSES)

        self.model: nn.Module = backbone

        # Cached collections
        self._bn_layers: Optional[List[nn.BatchNorm2d]] = None
        self._original_state: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # Forward                                                              #
    # ------------------------------------------------------------------ #

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)

    # ------------------------------------------------------------------ #
    # State management                                                     #
    # ------------------------------------------------------------------ #

    def save_original_state(self) -> None:
        """
        Deep-copy current model weights for later restoration.

        Call once after loading / fine-tuning.  The benchmark runner calls
        restore_original_state() before every new (corruption, method) pair
        to guarantee independent evaluation with no state leakage.
        """
        self._original_state = copy.deepcopy(self.model.state_dict())
        logger.debug("Model state snapshot saved.")

    def restore_original_state(self) -> None:
        """
        Restore model weights to the last saved snapshot.

        Raises
        ------
        RuntimeError
            If save_original_state() has not been called yet.
        """
        if self._original_state is None:
            raise RuntimeError(
                "No snapshot found. Call save_original_state() before adaptation."
            )
        self.model.load_state_dict(self._original_state)
        # Invalidate cached BN list (layers are replaced by load_state_dict)
        self._bn_layers = None
        logger.debug("Model state restored to snapshot.")

    # ------------------------------------------------------------------ #
    # BatchNorm utilities                                                  #
    # ------------------------------------------------------------------ #

    def get_batch_norm_layers(self) -> List[nn.BatchNorm2d]:
        """
        Return all BatchNorm2d layers in the model (cached after first call).

        Returns
        -------
        List[nn.BatchNorm2d]
            Ordered list of all BN layers (53 for ResNet-50).
        """
        if self._bn_layers is None:
            self._bn_layers = [
                m for m in self.model.modules()
                if isinstance(m, nn.BatchNorm2d)
            ]
        return self._bn_layers

    def reset_bn_running_stats(self) -> None:
        """
        Reset all BN running statistics (mean → 0, var → 1, num_batches → 0).

        Called by TTN before processing each corruption to ensure the BN stats
        reflect only the current test batch distribution, not prior batches.
        """
        for m in self.get_batch_norm_layers():
            m.reset_running_stats()

    def set_bn_training_mode(self, training: bool) -> None:
        """Switch all BN layers between train (stat update) and eval mode."""
        for m in self.get_batch_norm_layers():
            m.train(training)

    # ------------------------------------------------------------------ #
    # Gradient control                                                     #
    # ------------------------------------------------------------------ #

    def freeze_all_params(self) -> None:
        """
        Disable gradients for all parameters.

        Called by TENT before selectively re-enabling BN affine params (γ, β).
        """
        for p in self.model.parameters():
            p.requires_grad_(False)

    def unfreeze_bn_affine_params(self) -> None:
        """
        Re-enable gradients for BN scale (γ = weight) and shift (β = bias).

        Called immediately after freeze_all_params() during TENT setup so that
        only the lightweight affine parameters are updated during entropy
        minimisation — preventing catastrophic forgetting of feature weights.
        """
        for m in self.get_batch_norm_layers():
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)

    # ------------------------------------------------------------------ #
    # Entropy                                                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def get_entropy(logits: torch.Tensor) -> torch.Tensor:
        """
        Compute per-sample Shannon entropy from raw logits.

        H(p) = -Σ_c p_c · log(p_c),   where  p = softmax(logits)

        Unit: nats (natural log).  For C classes:
            H_min = 0       (all mass on one class — certain prediction)
            H_max = log(C)  (uniform over classes — maximum uncertainty)

        A 1e-8 epsilon prevents log(0) without meaningfully biasing entropy.

        Parameters
        ----------
        logits : torch.Tensor
            Shape (N, C) — raw pre-softmax model outputs.

        Returns
        -------
        torch.Tensor
            Shape (N,) — per-sample entropy in nats.
        """
        probs     = torch.softmax(logits, dim=1)
        log_probs = torch.log(probs + 1e-8)
        return -(probs * log_probs).sum(dim=1)

    def get_mean_entropy(self, logits: torch.Tensor) -> float:
        """Mean entropy (scalar) over a batch of logits."""
        return float(self.get_entropy(logits).mean().item())


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_model(
    weights_path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
    pretrained: bool = True,
) -> CIFAR10ResNet:
    """
    Build a CIFAR10ResNet and optionally load saved CIFAR-10 weights.

    The ImageNet-pretrained backbone is always loaded first (from torchvision
    cache or downloaded on first run).  If ``weights_path`` points to a valid
    checkpoint, the CIFAR-10 fine-tuned weights replace the ImageNet head.
    Otherwise, only the head is randomly initialised — useful for the demo
    where fine-tuning has not been run.

    After loading, ``save_original_state()`` is called automatically so the
    benchmark runner can reset the model before each adaptation method.

    Parameters
    ----------
    weights_path : str or None
        Path to a PyTorch checkpoint produced by ``fine_tune_on_cifar10()``.
        If None or the file does not exist, falls back to ImageNet backbone.
    device : torch.device
        Target device (cpu recommended for the full benchmark).

    Returns
    -------
    CIFAR10ResNet
        Model in eval mode with original state saved.
    """
    model = CIFAR10ResNet(pretrained=pretrained).to(device)

    if weights_path and os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device)
        # Support both plain state_dict and checkpoint dicts
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.model.load_state_dict(state_dict)
        logger.info("Loaded CIFAR-10 fine-tuned weights from %s", weights_path)
    else:
        if weights_path:
            logger.warning(
                "Checkpoint not found at '%s'. Using ImageNet backbone with "
                "randomly initialised FC head.  Run fine-tuning first for "
                "meaningful benchmark results.", weights_path
            )
        else:
            logger.info(
                "No weights_path provided — using ImageNet backbone + random head. "
                "Run fine-tuning for accurate benchmark results."
            )

    model.save_original_state()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def fine_tune_on_cifar10(
    model: CIFAR10ResNet,
    train_loader,
    device: torch.device = torch.device("cpu"),
    epochs: int = 10,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
    save_path: str = "./cifar10_resnet50.pth",
) -> CIFAR10ResNet:
    """
    Fine-tune the ResNet-50 head (layer4 + FC + BN) on CIFAR-10.

    Only the final residual block (layer4), the FC layer, and all BN
    affine parameters are updated.  Earlier backbone layers retain their
    ImageNet-pre-trained features.

    Parameters
    ----------
    model : CIFAR10ResNet
        Model to fine-tune (modified in-place).
    train_loader : DataLoader
        CIFAR-10 training set loader.
    device : torch.device
        Computation device.
    epochs : int
        Number of training epochs.
    lr : float
        Initial SGD learning rate.
    weight_decay : float
        L2 regularisation coefficient.
    save_path : str
        Path to save the fine-tuned checkpoint.

    Returns
    -------
    CIFAR10ResNet
        Fine-tuned model in eval mode, with original state re-saved.
    """
    logger.info("Starting CIFAR-10 fine-tuning for %d epochs …", epochs)

    # Selective unfreezing: layer4, FC, and BN layers
    for name, p in model.model.named_parameters():
        p.requires_grad_(
            "fc" in name
            or "layer4" in name
            or "bn" in name
            or "downsample.1" in name   # downsample BN layers
        )

    trainable = [p for p in model.model.parameters() if p.requires_grad]
    logger.info(
        "  Trainable parameters: %d / %d",
        sum(p.numel() for p in trainable),
        sum(p.numel() for p in model.model.parameters()),
    )

    optimizer = optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss  = 0.0
        total_corr  = 0
        total_n     = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                preds = model(inputs).argmax(dim=1)
                total_corr += preds.eq(targets).sum().item()
                total_n    += targets.size(0)

        scheduler.step()
        acc = total_corr / total_n
        logger.info(
            "  Epoch [%d/%d]  loss=%.4f  acc=%.2f%%",
            epoch, epochs, total_loss / len(train_loader), 100 * acc,
        )

    # Save checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.model.state_dict(),
            "epochs": epochs,
            "final_accuracy": acc,
        },
        save_path,
    )
    logger.info("Saved checkpoint to %s", save_path)

    model.save_original_state()
    model.eval()
    return model


def fine_tune_and_save(
    data_dir: str = "./data",
    save_path: str = "./cifar10_resnet50.pth",
    epochs: int = 10,
    device: Optional[torch.device] = None,
) -> None:
    """
    Convenience wrapper: download CIFAR-10, build model, fine-tune, save.

    Intended as a standalone script entry-point::

        python -c "
        from src.backbone.pretrained_model import fine_tune_and_save
        fine_tune_and_save('./data', './cifar10_resnet50.pth', epochs=10)
        "
    """
    from src.data.dataset_loader import get_cifar10_loaders

    if device is None:
        device = torch.device("cpu")

    train_loader, _ = get_cifar10_loaders(data_dir=data_dir, batch_size=128)
    model = build_model(device=device)
    fine_tune_on_cifar10(
        model, train_loader, device=device, epochs=epochs, save_path=save_path
    )


def evaluate_on_clean(
    model: CIFAR10ResNet,
    test_loader,
    device: torch.device = torch.device("cpu"),
) -> float:
    """
    Evaluate accuracy on clean CIFAR-10 test set.

    Parameters
    ----------
    model : CIFAR10ResNet
        Model to evaluate (will be set to eval mode).
    test_loader : DataLoader
        Clean CIFAR-10 test loader.
    device : torch.device
        Computation device.

    Returns
    -------
    float
        Top-1 accuracy in [0, 1].
    """
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            preds = model(inputs).argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            total   += targets.size(0)

    accuracy = correct / total
    logger.info("Clean CIFAR-10 accuracy: %.4f (%.2f%%)", accuracy, 100 * accuracy)
    return accuracy
