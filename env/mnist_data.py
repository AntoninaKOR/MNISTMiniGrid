"""Tiny self-contained MNIST loader used by the maze environment and the agent.

Downloads the official ubyte files into a local cache on first use. The
environment only needs the training split grouped by digit class; the agent's
classifier pre-training additionally uses the held-out 10k test split for
honest evaluation.
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mnist-maze"

# OSS mirror also used by PyTorch's torchvision.
_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_FILES = {
    "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz"),
    "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"),
}


def _download(filename: str, cache_dir: Path) -> Path:
    path = cache_dir / filename
    if not path.exists():
        cache_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MIRROR + filename, path)
    return path


def _parse_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051, f"bad magic in {path}: {magic}"
        buf = f.read()
    # ``frombuffer`` returns a read-only view of an immutable bytes object;
    # ``copy`` so downstream code (PyTorch, etc.) gets a writeable ndarray.
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols).copy()


def _parse_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, _ = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic in {path}: {magic}"
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8).copy()


def load_mnist(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    split: str = "train",
) -> tuple[np.ndarray, np.ndarray]:
    """Load one MNIST split as ``(images, labels)`` ``uint8`` arrays.

    ``split`` must be ``"train"`` (60k samples) or ``"test"`` (10k samples).
    """
    assert split in _FILES, f"unknown split {split!r}; expected one of {list(_FILES)}"
    cache_dir = Path(cache_dir)
    image_file, label_file = _FILES[split]
    images = _parse_images(_download(image_file, cache_dir))
    labels = _parse_labels(_download(label_file, cache_dir))
    return images, labels


def load_mnist_by_class(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    split: str = "train",
) -> list[np.ndarray]:
    """Load one MNIST split grouped by digit class.

    Returns a list of 10 ``(n_c, 28, 28)`` ``uint8`` arrays.
    """
    images, labels = load_mnist(cache_dir, split=split)
    return [np.ascontiguousarray(images[labels == c]) for c in range(10)]
