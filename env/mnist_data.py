"""Tiny self-contained MNIST loader used by the maze environment.

Downloads the official ubyte files into a local cache on first use and groups
them by digit class so the environment can sample a random image per class in
O(1).
"""

from __future__ import annotations

import gzip
import struct
import urllib.request
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = Path.home() / ".cache" / "mnist-maze"

_MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"
_IMAGE_FILE = "train-images-idx3-ubyte.gz"
_LABEL_FILE = "train-labels-idx1-ubyte.gz"


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
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def _parse_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, _ = struct.unpack(">II", f.read(8))
        assert magic == 2049, f"bad magic in {path}: {magic}"
        buf = f.read()
    return np.frombuffer(buf, dtype=np.uint8)


def load_mnist_by_class(
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
) -> list[np.ndarray]:
    """Load MNIST training images grouped by digit class.

    Returns a list of 10 ``(n_c, 28, 28)`` ``uint8`` arrays, one per class.
    """
    cache_dir = Path(cache_dir)
    images = _parse_images(_download(_IMAGE_FILE, cache_dir))
    labels = _parse_labels(_download(_LABEL_FILE, cache_dir))
    return [np.ascontiguousarray(images[labels == c]) for c in range(10)]
