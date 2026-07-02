"""
Mock corruption functions for unit tests and CI pipelines.

These functions apply simple, deterministic transformations to random
PyTorch tensors to simulate corruption effects without requiring any
file downloads.  They are intentionally lightweight — correctness of the
adaptation pipeline is validated, not photorealistic corruption fidelity.

Available corruptions
---------------------
gaussian_noise(tensor, severity)
    Adds scaled Gaussian noise proportional to severity.

blur(tensor, severity)
    Applies an average-pool blur with kernel size proportional to severity.

brightness(tensor, severity)
    Shifts pixel values by a constant proportional to severity.

Factory function
----------------
apply_mock_corruption(tensor, corruption_type, severity)
    Routes to the appropriate function by name.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Individual corruption functions
# ---------------------------------------------------------------------------


def gaussian_noise(
    tensor: torch.Tensor,
    severity: int = 3,
    seed: int = 0,
) -> torch.Tensor:
    """
    Add scaled Gaussian noise to an image tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Float image tensor of shape (N, C, H, W) or (C, H, W).
    severity : int
        Noise scale multiplier; severity 1 → σ=0.08, severity 5 → σ=0.40.
    seed : int
        Random seed for reproducibility in tests.

    Returns
    -------
    torch.Tensor
        Corrupted tensor of the same shape and dtype.
    """
    rng   = torch.Generator()
    rng.manual_seed(seed)
    sigma = 0.08 * severity
    noise = torch.randn_like(tensor, generator=rng) * sigma
    return tensor + noise


def blur(
    tensor: torch.Tensor,
    severity: int = 3,
) -> torch.Tensor:
    """
    Apply average-pool blur to simulate defocus or motion blur.

    The kernel size grows with severity:
        severity 1 → kernel 1×1 (identity)
        severity 2 → kernel 3×3
        severity 3 → kernel 5×5
        severity 4 → kernel 7×7
        severity 5 → kernel 9×9

    Parameters
    ----------
    tensor : torch.Tensor
        Float image tensor of shape (N, C, H, W).
        If 3-D (C, H, W), a batch dimension is added temporarily.
    severity : int
        Blur kernel size multiplier in [1, 5].

    Returns
    -------
    torch.Tensor
        Blurred tensor of the same shape.
    """
    kernel = max(1, severity * 2 - 1)
    if kernel == 1:
        return tensor

    squeeze = tensor.dim() == 3
    if squeeze:
        tensor = tensor.unsqueeze(0)

    pad     = kernel // 2
    blurred = F.avg_pool2d(tensor, kernel_size=kernel, stride=1, padding=pad)

    # Restore original spatial size if pool changed it
    if blurred.shape[-2:] != tensor.shape[-2:]:
        blurred = F.interpolate(
            blurred,
            size=tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    if squeeze:
        blurred = blurred.squeeze(0)

    return blurred


def brightness(
    tensor: torch.Tensor,
    severity: int = 3,
) -> torch.Tensor:
    """
    Increase image brightness by adding a constant offset.

    Parameters
    ----------
    tensor : torch.Tensor
        Float image tensor (any shape).
    severity : int
        Brightness shift magnitude: severity * 0.25 added to all pixels.

    Returns
    -------
    torch.Tensor
        Brightened tensor of the same shape.
    """
    shift = 0.25 * severity
    return tensor + shift


# ---------------------------------------------------------------------------
# Factory / dispatcher
# ---------------------------------------------------------------------------

_CORRUPTION_FN = {
    "gaussian_noise": gaussian_noise,
    "blur":           blur,
    "brightness":     brightness,
}


def apply_mock_corruption(
    tensor: torch.Tensor,
    corruption_type: str,
    severity: int = 3,
) -> torch.Tensor:
    """
    Apply a named mock corruption to a tensor.

    Parameters
    ----------
    tensor : torch.Tensor
        Input image tensor.
    corruption_type : str
        One of: 'gaussian_noise', 'blur', 'brightness'.
    severity : int
        Corruption severity in [1, 5].

    Returns
    -------
    torch.Tensor
        Corrupted tensor.

    Raises
    ------
    ValueError
        If ``corruption_type`` is not recognised.
    """
    if corruption_type not in _CORRUPTION_FN:
        raise ValueError(
            f"Unknown mock corruption '{corruption_type}'. "
            f"Valid: {sorted(_CORRUPTION_FN)}"
        )
    return _CORRUPTION_FN[corruption_type](tensor, severity=severity)


def make_random_batch(
    batch_size: int = 8,
    channels: int = 3,
    height: int = 32,
    width: int = 32,
    seed: int = 42,
) -> torch.Tensor:
    """
    Generate a repeatable random float32 image tensor.

    Used in unit tests that need a deterministic input.

    Parameters
    ----------
    batch_size : int
        Number of images.
    channels : int
        Number of channels (3 for RGB).
    height, width : int
        Spatial dimensions.
    seed : int
        Random seed.

    Returns
    -------
    torch.Tensor
        Float32 tensor of shape (batch_size, channels, height, width).
    """
    rng = torch.Generator()
    rng.manual_seed(seed)
    return torch.randn(batch_size, channels, height, width, generator=rng)
