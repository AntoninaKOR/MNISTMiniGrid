"""Tiny MNIST CNN used as the (frozen) observation encoder for the RL agent.

The classifier maps a single ``28x28`` grayscale image to 10 logits over the
digit classes. After pre-training, the RL pipeline uses ``argmax`` of those
logits as the discrete observation fed to :class:`agent.policy.GRUPolicy`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MNISTClassifier(nn.Module):
    """Small CNN: 2 conv blocks + 2 FC layers, ~210k parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Inputs: ``(B, 1, 28, 28)`` float in ``[0, 1]``; outputs 10-class logits."""
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    @torch.no_grad()
    def predict(self, images_uint8: torch.Tensor) -> torch.Tensor:
        """Convert ``(B, 28, 28) uint8`` (0..255) images to predicted digits."""
        x = images_uint8.to(torch.float32).div_(255.0).unsqueeze(1)
        return self.forward(x).argmax(dim=-1)


def load_classifier(
    checkpoint: str | Path,
    device: torch.device | str = "cpu",
) -> MNISTClassifier:
    """Load a pre-trained classifier from disk and put it in eval mode."""
    model = MNISTClassifier()
    state = torch.load(Path(checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def train_classifier(
    images: np.ndarray,
    labels: np.ndarray,
    *,
    epochs: int = 3,
    batch_size: int = 128,
    lr: float = 1e-3,
    val_fraction: float = 0.1,
    device: torch.device | str = "cpu",
    seed: int = 0,
    log_every: int = 100,
) -> tuple[MNISTClassifier, float]:
    """Train an :class:`MNISTClassifier` and return ``(model, val_accuracy)``."""
    assert images.ndim == 3 and images.shape[1:] == (28, 28)
    assert labels.shape == (images.shape[0],)

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(images.shape[0])
    n_val = int(round(images.shape[0] * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    device = torch.device(device)
    x_train = torch.from_numpy(images[train_idx]).to(device).float().div_(255.0).unsqueeze(1)
    y_train = torch.from_numpy(labels[train_idx]).to(device).long()
    x_val = torch.from_numpy(images[val_idx]).to(device).float().div_(255.0).unsqueeze(1)
    y_val = torch.from_numpy(labels[val_idx]).to(device).long()

    model = MNISTClassifier().to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    n_train = x_train.shape[0]
    step = 0
    for epoch in range(epochs):
        order = torch.randperm(n_train, device=device)
        for start in range(0, n_train, batch_size):
            batch = order[start : start + batch_size]
            logits = model(x_train[batch])
            loss = F.cross_entropy(logits, y_train[batch])
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if step % log_every == 0:
                acc = (logits.argmax(-1) == y_train[batch]).float().mean().item()
                print(f"epoch {epoch} step {step:>5d}  loss {loss.item():.4f}  train_acc {acc:.3f}")
            step += 1

    model.eval()
    with torch.no_grad():
        val_acc = (model(x_val).argmax(-1) == y_val).float().mean().item()
    print(f"validation accuracy: {val_acc:.4f}")
    return model, val_acc
