"""Recurrent actor-critic head for the PPO agent.

The policy expects already-decoded observations: a discrete digit class
(produced by the frozen MNIST classifier) plus the two-hot goal encoding.

Architecture: ``Embedding(n_digits, embed_dim) + Linear(goal_dim, embed_dim)
-> GRU(embed_dim * 2, hidden_dim) -> actor / critic heads``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Categorical


class GRUPolicy(nn.Module):
    """A small GRU-based actor-critic over discrete digit observations."""

    def __init__(
        self,
        n_digits: int,
        goal_dim: int,
        n_actions: int = 4,
        embed_dim: int = 32,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.digit_embed = nn.Embedding(n_digits, embed_dim)
        self.goal_proj = nn.Linear(goal_dim, embed_dim)
        self.gru = nn.GRU(embed_dim * 2, hidden_dim)
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def _encode(self, digit: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """Encode one or many ``(*, B)`` digit + ``(*, B, goal_dim)`` goal into ``(*, B, 2*emb)``."""
        return torch.cat([self.digit_embed(digit), self.goal_proj(goal)], dim=-1)

    def step(
        self,
        digit: torch.Tensor,
        goal: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One recurrent step. Returns ``(action_logits, value, new_hidden)``."""
        x = self._encode(digit, goal).unsqueeze(0)               # (1, B, 2*emb)
        out, new_hidden = self.gru(x, hidden.unsqueeze(0))       # (1, B, H), (1, B, H)
        h = out.squeeze(0)
        return self.actor(h), self.critic(h).squeeze(-1), new_hidden.squeeze(0)

    def evaluate(
        self,
        digits: torch.Tensor,
        goals: torch.Tensor,
        actions: torch.Tensor,
        episode_starts: torch.Tensor,
        initial_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate a stored ``(T, B)`` trajectory.

        Returns ``(log_probs, values, entropies)``, all shaped ``(T, B)``.
        ``episode_starts[t, b]`` is ``True`` when the GRU hidden state for
        env ``b`` should be reset to zero before processing step ``t``.

        Strategy: split the time axis at every step where *any* env resets,
        then call ``nn.GRU`` once per segment with the per-env reset mask
        applied to the hidden state at the segment boundary. Embedding /
        projection / actor / critic are all batched over the full trajectory.
        """
        T, B = digits.shape
        x = self._encode(digits, goals)  # (T, B, 2*emb)

        # Boundary timesteps: t where at least one env starts a new episode.
        # We always treat t = 0 as a boundary so the very first reset mask is
        # applied even if no env reset on step 0.
        boundary = episode_starts.any(dim=-1).clone()
        boundary[0] = True
        chunk_starts = boundary.nonzero(as_tuple=False).squeeze(-1).tolist()
        chunk_starts.append(T)

        hidden = initial_hidden
        hiddens = torch.empty(T, B, self.hidden_dim, device=digits.device, dtype=x.dtype)
        for i in range(len(chunk_starts) - 1):
            s, e = chunk_starts[i], chunk_starts[i + 1]
            if s >= e:
                continue
            mask = (~episode_starts[s]).to(x.dtype).unsqueeze(-1)
            hidden = hidden * mask
            chunk_out, h_out = self.gru(x[s:e], hidden.unsqueeze(0))
            hiddens[s:e] = chunk_out
            hidden = h_out.squeeze(0)

        logits = self.actor(hiddens)                      # (T, B, n_actions)
        values = self.critic(hiddens).squeeze(-1)         # (T, B)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        return log_probs, values, entropies
