"""Recurrent actor-critic head for the PPO agent.

The policy expects already-decoded observations: a discrete digit class
(produced by the frozen MNIST classifier) plus the two-hot goal encoding.
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
        self.gru = nn.GRUCell(embed_dim * 2, hidden_dim)
        self.actor = nn.Linear(hidden_dim, n_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(
        self,
        digit: torch.Tensor,
        goal: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One recurrent step. Returns ``(action_logits, value, new_hidden)``."""
        d = self.digit_embed(digit)
        g = self.goal_proj(goal)
        x = torch.cat([d, g], dim=-1)
        h = self.gru(x, hidden)
        return self.actor(h), self.critic(h).squeeze(-1), h

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
        """
        T, B = digits.shape
        hidden = initial_hidden
        log_probs = torch.empty(T, B, device=digits.device)
        values = torch.empty(T, B, device=digits.device)
        entropies = torch.empty(T, B, device=digits.device)
        for t in range(T):
            mask = (~episode_starts[t]).float().unsqueeze(-1)
            hidden = hidden * mask
            logits, value, hidden = self.step(digits[t], goals[t], hidden)
            dist = Categorical(logits=logits)
            log_probs[t] = dist.log_prob(actions[t])
            entropies[t] = dist.entropy()
            values[t] = value
        return log_probs, values, entropies
