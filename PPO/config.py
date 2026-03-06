"""
Hyperparameters for PPO training on Checkers (deepdraughts).
"""

import os
from dataclasses import dataclass, field

def _auto_n_envs() -> int:
    """Use all CPU cores minus 2 (for main process + OS). Minimum 2."""
    return max(os.cpu_count() - 2, 2)

def _auto_rollout_steps(n_envs: int, target: int = 16384) -> int:
    """Round target up to nearest multiple of n_envs."""
    return ((target + n_envs - 1) // n_envs) * n_envs

@dataclass
class PPOConfig:
    # --- Environment ---
    board_size: int = 8          # 8x8 board
    n_actions: int = 280         # all possible moves on 8x8 board
    vec_board_channels: int = 4  # white_men, white_king, black_men, black_king
    vec_state_dim: int = 19      # N_STATE_64 = 8*2 + 3

    # --- Network architecture ---
    conv_filters: list = field(default_factory=lambda: [32, 64, 128])
    fc_hidden: int = 256
    state_hidden: int = 64
    n_res_blocks: int = 4        # number of residual blocks in conv backbone

    # --- PPO hyperparameters ---
    lr: float = 3e-4
    gamma: float = 0.99          # discount factor
    gae_lambda: float = 0.95     # balance between Monte Carlo (1) and TD (0)
    clip_eps: float = 0.2        # how much can policy deviate from old one in update
    entropy_coef: float = 0.05   # entropy bonus start (higher → more exploration)
    entropy_coef_end: float = 0.01  # entropy bonus end (decays linearly during training)
    value_coef: float = 0.5      # value loss coefficient
    max_grad_norm: float = 0.5   # gradient clipping
    lr_min_fraction: float = 0.1 # minimum LR as fraction of initial (floor for linear decay)

    # --- Training ---
    n_epochs: int = 4            # PPO epochs per update
    batch_size: int = 256        # mini-batch size (bigger → more stable gradients)
    rollout_steps: int = -1      # steps per rollout (-1 = auto: ~8192 rounded to n_envs)
    n_envs: int = -1              # parallel workers (-1 = auto: cpu_count - 2)
    total_timesteps: int = 10_000_000  # total training timesteps
    opponent_pool_size: int = 60     # max past model snapshots to keep
    opponent_pool_interval: int = 25 # add current model to pool every N updates
    mcts_opponent_ratio: float = 0.15 # fraction of games vs MCTS (0.0 = pure self-play, 1.0 = pure MCTS)
    mcts_opponent_playouts: int = 500 # MCTS playouts per move during training

    # --- Rewards ---
    reward_win: float = 1.0
    reward_loss: float = -1.0
    reward_draw: float = 0.0
    reward_step: float = -0.001  # small penalty per step to encourage faster play
    reward_capture: float = 0.02 # small reward for captures (too high → greedy/tactical play)
    max_game_steps: int = 300    # max steps before forced draw

    # --- Logging & checkpoints ---
    log_interval: int = 10       # log every N updates
    save_interval: int = 100     # save model every N updates
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # --- Device ---
    use_gpu: bool = True

    def __post_init__(self):
        if self.n_envs <= 0:
            self.n_envs = _auto_n_envs()
        if self.rollout_steps <= 0:
            self.rollout_steps = _auto_rollout_steps(self.n_envs)
