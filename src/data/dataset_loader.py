"""
Dataset loading utilities for CIFAR-10 (source domain) and CIFAR-10-C
(target domain, 15 corruptions × 5 severities).

Usage
-----
# Real CIFAR-10-C (requires download):
    loader = CIFAR10CLoader(data_dir="./CIFAR-10-C")
    dl = loader.get_loader("gaussian_noise", severity=3)

# Mock loader (no download, for CI / demo):
    mock = MockCorruptionLoader(batch_size=64, num_samples=640)
    dl = mock.get_loader("gaussian_noise", severity=3)

# Clean CIFAR-10 train/test:
    train_dl, test_dl = get_cifar10_loaders(data_dir="./data")

CIFAR-10-C file format
----------------------
Each .npy file: shape (50000, 32, 32, 3), uint8.
Five severity levels stacked: indices 0–9999 = severity 1, 10000–19999 = severity 2, …
labels.npy: shape (50000,), int64. Shared across all corruption types.

Download: https://zenodo.org/record/2535967  (CIFAR-10-C.tar, ~300 MB)
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

NUM_CLASSES           = 10
SAMPLES_PER_SEVERITY  = 10_000   # 10 000 images per severity level
TOTAL_SAMPLES         = 50_000   # 5 severity levels × 10 000

CORRUPTION_CATEGORIES = {
    "noise":   ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur":    ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

CORRUPTION_TYPES: List[str] = [
    c for cats in CORRUPTION_CATEGORIES.values() for c in cats
]

# Corruptions available in the mock loader (used by demo / CI)
MOCK_CORRUPTION_TYPES = ["gaussian_noise", "blur", "brightness"]


def _normalize_tensor(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of CIFAR-10 images with channel-wise mean/std.

    Parameters
    ----------
    images : torch.Tensor
        Shape (N, 3, 32, 32), float32 in [0, 1].

    Returns
    -------
    torch.Tensor
        Normalized images.
    """
    mean = torch.tensor(CIFAR10_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD).view(1, 3, 1, 1)
    return (images - mean) / std


# ---------------------------------------------------------------------------
# Clean CIFAR-10
# ---------------------------------------------------------------------------


def get_cifar10_loaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Return (train_loader, test_loader) for CIFAR-10.

    The test loader uses only ToTensor + Normalize (no augmentation).
    The training loader adds random crop and horizontal flip.

    Parameters
    ----------
    data_dir : str
        Root directory where CIFAR-10 will be downloaded / cached.
    batch_size : int
        Mini-batch size for both loaders.
    num_workers : int
        Number of worker processes for DataLoader.

    Returns
    -------
    Tuple[DataLoader, DataLoader]
        (train_loader, test_loader)
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, test_loader


# ---------------------------------------------------------------------------
# CIFAR-10-C (real dataset)
# ---------------------------------------------------------------------------


def _normalize_tensor(images: torch.Tensor) -> torch.Tensor:
    """
    Normalize an image tensor with CIFAR-10 channel statistics.

    Parameters
    ----------
    images : torch.Tensor
        Float tensor of shape (N, 3, H, W) in range [0, 1].
    """
    mean = torch.tensor(CIFAR10_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
    std  = torch.tensor(CIFAR10_STD,  dtype=torch.float32).view(1, 3, 1, 1)
    return (images - mean) / std


class CIFAR10CLoader:
    """
    Loads CIFAR-10-C corruption data from numpy arrays on disk.

    Parameters
    ----------
    data_dir : str
        Directory containing the extracted CIFAR-10-C .npy files.
        Expected layout::

            CIFAR-10-C/
                gaussian_noise.npy   (50000, 32, 32, 3)  uint8
                shot_noise.npy
                ...
                labels.npy           (50000,)             int64

    batch_size : int
        Mini-batch size for the returned DataLoaders.
    """

    def __init__(self, data_dir: str, batch_size: int = 128) -> None:
        self.data_dir   = data_dir
        self.batch_size = batch_size
        self._labels: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def get_loader(
        self,
        corruption_type: str,
        severity: int,
    ) -> DataLoader:
        """
        Return a DataLoader for one corruption type at one severity level.

        Parameters
        ----------
        corruption_type : str
            One of the 15 corruption types in CORRUPTION_TYPES.
        severity : int
            Integer in [1, 5].

        Returns
        -------
        DataLoader
            Yields (image_tensor, label_tensor) pairs.
            Images are float32 in the normalized CIFAR-10 space.
        """
        self._validate(corruption_type, severity)
        images_np = self._load_images(corruption_type, severity)
        labels_np = self._load_labels(severity)

        images_tensor = (
            torch.from_numpy(images_np)
            .float()
            .permute(0, 3, 1, 2)
            .div_(255.0)
        )
        images_tensor = _normalize_tensor(images_tensor)
        labels_tensor = torch.from_numpy(labels_np).long()

        dataset = TensorDataset(images_tensor, labels_tensor)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def available_corruptions(self) -> List[str]:
        """Return list of corruption types for which .npy files exist."""
        return [
            c for c in CORRUPTION_TYPES
            if os.path.exists(os.path.join(self.data_dir, f"{c}.npy"))
        ]

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _validate(self, corruption_type: str, severity: int) -> None:
        if corruption_type not in CORRUPTION_TYPES:
            raise ValueError(
                f"Unknown corruption '{corruption_type}'. "
                f"Valid types:\n  {CORRUPTION_TYPES}"
            )
        if not 1 <= severity <= 5:
            raise ValueError(
                f"Severity must be in [1, 5], got {severity}."
            )
        npy_path = os.path.join(self.data_dir, f"{corruption_type}.npy")
        if not os.path.exists(npy_path):
            raise FileNotFoundError(
                f"CIFAR-10-C file not found: {npy_path}\n"
                "Download from: https://zenodo.org/record/2535967"
            )

    def _load_images(self, corruption_type: str, severity: int) -> np.ndarray:
        start = (severity - 1) * SAMPLES_PER_SEVERITY
        end   = severity * SAMPLES_PER_SEVERITY
        npy_path = os.path.join(self.data_dir, f"{corruption_type}.npy")
        return np.load(npy_path, mmap_mode="r")[start:end]  # (10000, 32, 32, 3)

    def _load_labels(self, severity: int) -> np.ndarray:
        if self._labels is None:
            label_path = os.path.join(self.data_dir, "labels.npy")
            self._labels = np.load(label_path)
        start = (severity - 1) * SAMPLES_PER_SEVERITY
        end   = severity * SAMPLES_PER_SEVERITY
        return self._labels[start:end]  # (10000,)


# ---------------------------------------------------------------------------
# Kaggle CIFAR-10 loader (Python pickle batch format)
# ---------------------------------------------------------------------------


class KaggleCIFAR10Loader:
    """
    Load CIFAR-10 from the standard Kaggle Python pickle format.

    The Kaggle CIFAR-10 dataset (``cifar-10-python``) distributes the data
    as Python pickle files identical to the original format from
    cs.toronto.edu::

        cifar-10-batches-py/
            data_batch_1       (10 000 images each)
            data_batch_2
            data_batch_3
            data_batch_4
            data_batch_5
            test_batch
            batches.meta

    This loader reads those files and returns DataLoaders compatible with
    the benchmark pipeline.

    Parameters
    ----------
    data_dir : str
        Directory containing the ``cifar-10-batches-py/`` folder,
        or the folder itself.
    batch_size : int
        Mini-batch size for the returned DataLoaders.

    Example
    -------
    ::

        loader = KaggleCIFAR10Loader("./cifar-10-python")
        train_dl, test_dl = loader.get_loaders()
    """

    def __init__(self, data_dir: str, batch_size: int = 128) -> None:
        self.data_dir   = data_dir
        self.batch_size = batch_size
        # Auto-detect nested folder
        nested = os.path.join(data_dir, "cifar-10-batches-py")
        if os.path.isdir(nested):
            self._root = nested
        else:
            self._root = data_dir

    def _unpickle(self, filepath: str) -> dict:
        """Read a single CIFAR-10 batch file."""
        import pickle
        with open(filepath, "rb") as f:
            batch = pickle.load(f, encoding="bytes")
        return batch

    def _load_batch(self, filepath: str):
        """Load images and labels from one pickle batch file."""
        batch  = self._unpickle(filepath)
        images = batch[b"data"]                  # (N, 3072) uint8
        labels = batch.get(b"labels", batch.get(b"fine_labels", []))
        images = images.reshape(-1, 3, 32, 32)   # (N, 3, 32, 32)
        return images, np.array(labels)

    def get_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Return (train_loader, test_loader) from Kaggle CIFAR-10 pickles.

        The images are normalised with CIFAR-10 statistics.

        Returns
        -------
        Tuple[DataLoader, DataLoader]
            Train and test DataLoaders.
        """
        # Load training batches
        all_images = []
        all_labels = []
        for i in range(1, 6):
            path = os.path.join(self._root, f"data_batch_{i}")
            if os.path.exists(path):
                imgs, lbls = self._load_batch(path)
                all_images.append(imgs)
                all_labels.append(lbls)

        if not all_images:
            raise FileNotFoundError(
                f"No CIFAR-10 batch files found in {self._root}. "
                "Expected files named data_batch_1 ... data_batch_5."
            )

        train_images = np.concatenate(all_images, axis=0)
        train_labels = np.concatenate(all_labels, axis=0)

        # Load test batch
        test_path = os.path.join(self._root, "test_batch")
        if os.path.exists(test_path):
            test_images, test_labels = self._load_batch(test_path)
        else:
            raise FileNotFoundError(
                f"Test batch not found at {test_path}."
            )

        # Convert to tensors and normalise
        train_t = _normalize_tensor(
            torch.from_numpy(train_images).float().div_(255.0)
        )
        test_t = _normalize_tensor(
            torch.from_numpy(test_images).float().div_(255.0)
        )
        train_l = torch.from_numpy(train_labels).long()
        test_l = torch.from_numpy(test_labels).long()

        train_loader = DataLoader(
            TensorDataset(train_t, train_l),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            TensorDataset(test_t, test_l),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        logger.info(
            "KaggleCIFAR10Loader: %d train, %d test samples loaded.",
            len(train_labels), len(test_labels),
        )
        return train_loader, test_loader


# ---------------------------------------------------------------------------
# Mock loader (CI / demo — no download required)
# ---------------------------------------------------------------------------


def _apply_mock_blur(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    """Average-pool blur as a cheap mock for defocus/motion blur."""
    if kernel_size <= 1:
        return images
    pad = kernel_size // 2
    blurred = F.avg_pool2d(images, kernel_size=kernel_size, stride=1, padding=pad)
    if blurred.shape[-2:] != images.shape[-2:]:
        blurred = F.interpolate(
            blurred, size=images.shape[-2:], mode="bilinear", align_corners=False
        )
    return blurred


class MockCorruptionLoader:
    """
    Synthetic corruption loader that requires no file downloads.

    Generates random image tensors and applies simple transformations to
    simulate three representative corruption types:
        - gaussian_noise: additive Gaussian noise scaled by severity
        - blur:           average-pool blur with kernel proportional to severity
        - brightness:     additive brightness shift

    Used for CI pipelines and the ``python examples/run_demo.py`` demo.
    Results are not scientifically meaningful (random labels), but the full
    adaptation pipeline executes identically to the real benchmark.

    Parameters
    ----------
    batch_size : int
        Mini-batch size for the returned DataLoaders.
    num_samples : int
        Total number of synthetic samples per DataLoader.
    seed : int
        Random seed for reproducibility of synthetic data.
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_samples: int = 640,
        seed: int = 42,
    ) -> None:
        self.batch_size  = batch_size
        self.num_samples = num_samples
        self.seed        = seed

    def get_loader(
        self,
        corruption_type: str,
        severity: int,
        num_classes: int = NUM_CLASSES,
    ) -> DataLoader:
        """
        Return a DataLoader with synthetic corrupted data.

        Parameters
        ----------
        corruption_type : str
            'gaussian_noise', 'blur', or 'brightness'.
            Any other value defaults to Gaussian noise.
        severity : int
            Corruption intensity, integer in [1, 5].
        num_classes : int
            Number of output classes (default 10 for CIFAR-10).
        """
        rng = torch.Generator()
        rng.manual_seed(self.seed + severity)

        # Random base images (already approximately normalised)
        images = torch.randn(self.num_samples, 3, 32, 32, generator=rng)

        if corruption_type == "gaussian_noise":
            noise_std = 0.15 * severity
            images = images + torch.randn_like(images) * noise_std
        elif corruption_type == "blur":
            kernel_size = max(2, severity * 2 - 1)   # 1, 3, 5, 7, 9
            images = _apply_mock_blur(images, kernel_size)
        elif corruption_type == "brightness":
            brightness_shift = 0.25 * severity
            images = images + brightness_shift
        else:
            # Fallback: generic scaled noise
            images = images + torch.randn_like(images) * (0.08 * severity)

        labels = torch.randint(0, num_classes, (self.num_samples,), generator=rng)

        dataset = TensorDataset(images, labels)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )
