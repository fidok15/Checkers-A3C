"""
Implements:
  - Rollout buffer for storing transitions
  - GAE (Generalized Advantage Estimation)
  - Clipped surrogate objective
  - Mini-batch updates
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional

from config import PPOConfig
from model import ActorCritic


class RolloutBuffer:

    def __init__(self, config: PPOConfig, device: torch.device):
        self.cfg = config
        self.device = device
        self.size = config.rollout_steps
        self.pos = 0

        # Pre-allocate tensors
        bs = config.board_size
        self.vec_boards = np.zeros((self.size, 4, bs, bs), dtype=np.float32)
        self.vec_states = np.zeros((self.size, config.vec_state_dim), dtype=np.float32)
        self.action_masks = np.zeros((self.size, config.n_actions), dtype=np.float32)
        self.actions = np.zeros(self.size, dtype=np.int64)
        self.log_probs = np.zeros(self.size, dtype=np.float32)
        self.rewards = np.zeros(self.size, dtype=np.float32)
        self.values = np.zeros(self.size, dtype=np.float32)
        self.dones = np.zeros(self.size, dtype=np.float32)

        # Computed after rollout
        self.advantages = np.zeros(self.size, dtype=np.float32)
        self.returns = np.zeros(self.size, dtype=np.float32)

    def add(self, vec_board, vec_state, action_mask, action, log_prob, reward, value, done):
        idx = self.pos
        self.vec_boards[idx] = vec_board
        self.vec_states[idx] = vec_state
        self.action_masks[idx] = action_mask
        self.actions[idx] = action
        self.log_probs[idx] = log_prob
        self.rewards[idx] = reward
        self.values[idx] = value
        self.dones[idx] = float(done)
        self.pos += 1

    def is_full(self) -> bool:
        return self.pos >= self.size

    def reset(self):
        self.pos = 0

    def compute_gae(self, last_value: float, last_done: bool):
        """
        Standard single-player GAE.

        With the opponent pool, all buffer entries are from the LEARNER's
        perspective.  Consecutive non-terminal entries are consecutive
        learner turns (the opponent moved in between as part of the
        environment dynamics).  No negation is needed.
        """
        gamma = self.cfg.gamma
        lam = self.cfg.gae_lambda
        n = self.pos

        last_gae = 0.0
        for t in reversed(range(n)):
            if self.dones[t]:
                # Terminal step – no next state to bootstrap from.
                delta = self.rewards[t] - self.values[t]
                last_gae = delta
            else:
                if t == n - 1:
                    next_value = 0.0 if last_done else last_value
                else:
                    next_value = self.values[t + 1]

                delta = self.rewards[t] + gamma * next_value - self.values[t]
                last_gae = delta + gamma * lam * last_gae

            self.advantages[t] = last_gae

        self.returns[:n] = self.advantages[:n] + self.values[:n]

    def get_batches(self, batch_size: int):
        """Yield mini-batch indices (shuffled)."""
        n = self.pos
        indices = np.arange(n)
        np.random.shuffle(indices)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            yield indices[start:end]

    def to_tensors(self, indices):
        """Convert a batch of indices to GPU/CPU tensors."""
        return (
            torch.tensor(self.vec_boards[indices], device=self.device),
            torch.tensor(self.vec_states[indices], device=self.device),
            torch.tensor(self.action_masks[indices], device=self.device),
            torch.tensor(self.actions[indices], device=self.device),
            torch.tensor(self.log_probs[indices], device=self.device),
            torch.tensor(self.returns[indices], device=self.device),
            torch.tensor(self.advantages[indices], device=self.device),
        )


class PPO:

    def __init__(self, config: PPOConfig):
        self.cfg = config
        self.device = torch.device(
            "cuda" if config.use_gpu and torch.cuda.is_available() else "cpu"
        )

        # Actor-Critic network
        self.model = ActorCritic(config).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config.lr, eps=1e-5
        )

        # Rollout buffer
        self.buffer = RolloutBuffer(config, self.device)

        # Stats
        self.update_count = 0

    @torch.no_grad()
    def select_action(self, vec_board, vec_state, action_mask):
        """
        Select an action given observation + mask (single step, no grad).

        Returns: action (int), log_prob (float), value (float)
        """
        vb = torch.tensor(vec_board, device=self.device).unsqueeze(0)
        vs = torch.tensor(vec_state, device=self.device).unsqueeze(0)
        am = torch.tensor(action_mask, device=self.device).unsqueeze(0)

        action, log_prob, _, value = self.model.get_action_and_value(vb, vs, am)

        return (
            action.item(),
            log_prob.item(),
            value.item(),
        )

    @torch.no_grad()
    def estimate_value(self, vec_board, vec_state, action_mask):
        """Get V(s) for bootstrapping."""
        vb = torch.tensor(vec_board, device=self.device).unsqueeze(0)
        vs = torch.tensor(vec_state, device=self.device).unsqueeze(0)
        am = torch.tensor(action_mask, device=self.device).unsqueeze(0)
        return self.model.get_value(vb, vs, am).item()

    def update(self):
        """Run PPO update on the filled buffer. Returns dict of losses."""
        cfg = self.cfg
        self.model.train()

        total_pg_loss = 0.0
        total_v_loss = 0.0
        total_entropy = 0.0
        total_loss = 0.0
        n_updates = 0

        for epoch in range(cfg.n_epochs):
            for batch_idx in self.buffer.get_batches(cfg.batch_size):
                (
                    b_boards, b_states, b_masks,
                    b_actions, b_old_logprobs,
                    b_returns, b_advantages,
                ) = self.buffer.to_tensors(batch_idx)

                # Normalize advantages
                b_advantages = (b_advantages - b_advantages.mean()) / (
                    b_advantages.std() + 1e-8
                )

                # Forward pass
                _, new_log_probs, entropy, new_values = self.model.get_action_and_value(
                    b_boards, b_states, b_masks, b_actions
                )

                # PPO clipped surrogate loss
                ratio = torch.exp(new_log_probs - b_old_logprobs)
                surr1 = ratio * b_advantages
                surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * b_advantages
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                v_loss = F.mse_loss(new_values, b_returns)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + cfg.value_coef * v_loss + cfg.entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                total_pg_loss += pg_loss.item()
                total_v_loss += v_loss.item()
                total_entropy += (-entropy_loss).item()
                total_loss += loss.item()
                n_updates += 1

        self.update_count += 1
        self.buffer.reset()

        return {
            "policy_loss": total_pg_loss / max(n_updates, 1),
            "value_loss": total_v_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
            "total_loss": total_loss / max(n_updates, 1),
        }

    def save(self, path: str, total_steps: int = 0):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "update_count": self.update_count,
                "total_steps": total_steps,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.update_count = checkpoint.get("update_count", 0)
        return checkpoint.get("total_steps", 0)


# Need F for value loss
import torch.nn.functional as F
