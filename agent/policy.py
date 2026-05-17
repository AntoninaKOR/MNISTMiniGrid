"""Recurrent actor-critic head for the PPO agent.

The policy expects already-decoded observations: a discrete digit class
(produced by the frozen MNIST classifier), the two-hot goal encoding, and
the previous discrete action (or a dedicated ``"no action yet"`` token at
the very first step of an episode). 

Architecture:
    Embedding(n_digits, embed) + Linear(goal_dim, embed) + Embedding(n_actions + 1, embed)
    -> GRU(embed * 3, hidden, num_layers) -> actor / critic heads
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
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        assert num_layers >= 1
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_actions = n_actions
        self.digit_embed = nn.Embedding(n_digits, n_digits)
        n_action_tokens = n_actions + 1  # +1 for "no previous action" / start of episode
        self.action_embed = nn.Embedding(n_action_tokens, n_action_tokens)
        self.goal_proj = nn.Linear(goal_dim, embed_dim)
        self.gru = nn.GRU(
            n_digits + n_action_tokens + embed_dim, hidden_dim, num_layers=num_layers
        )
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, n_actions),
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return zero hidden state of shape ``(num_layers, batch, hidden_dim)``."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def initial_prev_action(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return the dedicated 'no previous action' token for each env."""
        return torch.full((batch_size,), self.n_actions, dtype=torch.long, device=device)

    def _encode(
        self,
        digit: torch.Tensor,
        goal: torch.Tensor,
        prev_action: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                self.digit_embed(digit),
                self.goal_proj(goal),
                self.action_embed(prev_action),
            ],
            dim=-1,
        )

    def step(
        self,
        digit: torch.Tensor,
        goal: torch.Tensor,
        prev_action: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One recurrent step. Returns ``(action_logits, value, new_hidden)``.

        ``hidden`` has shape ``(num_layers, B, hidden_dim)``; the returned
        ``new_hidden`` has the same shape. The actor/critic read the *top*
        layer's output.
        """
        x = self._encode(digit, goal, prev_action).unsqueeze(0)  # (1, B, 3*emb)
        out, new_hidden = self.gru(x, hidden)                    # (1, B, H), (L, B, H)
        h = out.squeeze(0)                                       # (B, H) -- top-layer output
        return self.actor(h), self.critic(h).squeeze(-1), new_hidden

    def evaluate(
        self,
        digits: torch.Tensor,
        goals: torch.Tensor,
        prev_actions: torch.Tensor,
        actions: torch.Tensor,
        episode_starts: torch.Tensor,
        initial_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Re-evaluate a stored ``(T, B)`` trajectory.

        Returns ``(log_probs, values, entropies)``, all shaped ``(T, B)``.
        ``episode_starts[t, b]`` is ``True`` when the GRU hidden state for
        env ``b`` should be reset to zero before processing step ``t``.
        ``initial_hidden`` has shape ``(num_layers, B, hidden_dim)``.
        """
        T, B = digits.shape
        x = self._encode(digits, goals, prev_actions)  # (T, B, 3*emb)

        # Boundary timesteps: t where at least one env starts a new episode.
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
            # ``episode_starts[s]`` is (B,) bool -> broadcast over (L, B, H).
            mask = (~episode_starts[s]).to(x.dtype).view(1, -1, 1)
            hidden = hidden * mask
            chunk_out, hidden = self.gru(x[s:e], hidden)  # (chunk, B, H), (L, B, H)
            hiddens[s:e] = chunk_out

        logits = self.actor(hiddens)                      # (T, B, n_actions)
        values = self.critic(hiddens).squeeze(-1)         # (T, B)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropies = dist.entropy()
        return log_probs, values, entropies
