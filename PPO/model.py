"""
Actor-Critic network for PPO on Checkers.

Architecture:
  - Shared convolutional backbone processes the 4x8x8 board tensor.
  - A small MLP encodes the auxiliary state vector (19-dim).
  - Their outputs are concatenated and fed into two heads:
      • Policy head  → logits over 280 actions (masked before softmax)
      • Value head   → scalar V(s)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import PPOConfig


class ResidualBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


class ActorCritic(nn.Module):
    """
    Input:
        vec_board : (B, 4, 8, 8)  – board channels
        vec_state : (B, 19)       – auxiliary state features
        action_mask : (B, 280)    – 1 for legal actions, 0 for illegal

    Output:
        policy : Categorical distribution over legal actions
        value  : (B, 1) estimated state value
    """

    def __init__(self, config: PPOConfig | None = None):
        super().__init__()
        cfg = config or PPOConfig()
        self.n_actions = cfg.n_actions

        # --- Board encoder (conv backbone) ---
        filters = cfg.conv_filters  # [32, 64, 128]
        layers = []
        in_ch = cfg.vec_board_channels  # 4
        for out_ch in filters:
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            in_ch = out_ch
        self.conv_backbone = nn.Sequential(*layers)

        n_res = getattr(cfg, 'n_res_blocks', 4)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(filters[-1]) for _ in range(n_res)]
        )

        # 1×1 conv to reduce channels before flatten (preserves spatial info)
        reduce_ch = 32
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(filters[-1], reduce_ch, kernel_size=1),
            nn.BatchNorm2d(reduce_ch),
            nn.ReLU(inplace=True),
        )
        conv_flat = reduce_ch * cfg.board_size * cfg.board_size  # 32*8*8 = 2048

        # --- State encoder ---
        self.state_encoder = nn.Sequential(
            nn.Linear(cfg.vec_state_dim, cfg.state_hidden),
            nn.ReLU(),
            nn.Linear(cfg.state_hidden, cfg.state_hidden),
            nn.ReLU(),
        )

        combined_dim = conv_flat + cfg.state_hidden  # 1024 + 64 = 1088

        # --- Policy head ---
        self.policy_head = nn.Sequential(
            nn.Linear(combined_dim, cfg.fc_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fc_hidden, cfg.n_actions),
        )

        # --- Value head ---
        self.value_head = nn.Sequential(
            nn.Linear(combined_dim, cfg.fc_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fc_hidden, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)

        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.zeros_(self.policy_head[-1].bias)
        
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
        nn.init.zeros_(self.value_head[-1].bias)

    def forward(self, vec_board, vec_state, action_mask=None):
        """
        Args:
            vec_board:   (B, 4, 8, 8) float tensor
            vec_state:   (B, 19) float tensor
            action_mask: (B, 280) float tensor – 1 for legal, 0 for illegal

        Returns:
            logits: (B, 280) raw policy logits (masked)
            value:  (B, 1)   state value estimate
        """
        # Board pathway
        x = self.conv_backbone(vec_board)
        x = self.res_blocks(x)
        x = self.conv_reduce(x)
        x = x.view(x.size(0), -1)  # flatten (B, 16*8*8)

        # State pathway
        s = self.state_encoder(vec_state)

        # Combine
        combined = torch.cat([x, s], dim=-1)

        # Heads
        logits = self.policy_head(combined)
        value = self.value_head(combined)

        # Mask illegal actions with large negative logits
        if action_mask is not None:
            logits = logits - (1.0 - action_mask) * 1e8

        return logits, value

    def get_action_and_value(self, vec_board, vec_state, action_mask, action=None):
        """
        Used during rollout collection and PPO update.

        Returns:
            action, log_prob, entropy, value
        """
        logits, value = self.forward(vec_board, vec_state, action_mask)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, log_prob, entropy, value.squeeze(-1)

    def get_value(self, vec_board, vec_state, action_mask=None):
        """Return V(s) only – used for bootstrapping."""
        _, value = self.forward(vec_board, vec_state, action_mask)
        return value.squeeze(-1)

















