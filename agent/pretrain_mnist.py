"""Pre-train an MNIST classifier and save its checkpoint.

The resulting checkpoint is loaded by :mod:`agent.train` as the (frozen)
observation encoder for the PPO agent.

Example:
    python -m agent.pretrain_mnist --epochs 3 \
        --output checkpoints/mnist_classifier.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from agent.mnist_classifier import train_classifier
from env.mnist_data import load_mnist_by_class


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--mnist-cache", type=Path, default=Path(".mnist_cache"))
    p.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/mnist_classifier.pt"),
        help="Where to save the trained classifier state_dict.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    banks = load_mnist_by_class(cache_dir=args.mnist_cache)
    images = np.concatenate(banks, axis=0)
    labels = np.concatenate([np.full(b.shape[0], c, dtype=np.int64) for c, b in enumerate(banks)])

    model, val_acc = train_classifier(
        images,
        labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"Saved classifier (val acc {val_acc:.4f}) to {args.output}")


if __name__ == "__main__":
    main()
