"""
Hyperparameters for PPO training on Checkers (deepdraughts).
"""

from dataclasses import dataclass, field


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

    # --- PPO hyperparameters ---
    lr: float = 3e-4
    gamma: float = 0.99          # discount factor
    gae_lambda: float = 0.95     # balance between Monte Carlo (1) and TD (0)
    clip_eps: float = 0.2        # how much can policy deviate from old one in update
    entropy_coef: float = 0.03   # entropy bonus coefficient (higher → more exploration)
    value_coef: float = 0.5      # value loss coefficient
    max_grad_norm: float = 0.5   # gradient clipping

    # --- Training ---
    n_epochs: int = 4            # PPO epochs per update
    batch_size: int = 128        # mini-batch size (bigger → more stable gradients)
    rollout_steps: int = 2048    # steps per rollout before update (more games per update)
    n_envs: int = 1              # number of parallel self-play environments
    total_timesteps: int = 2_000_000  # total training timesteps
    opponent_pool_size: int = 20     # max past model snapshots to keep
    opponent_pool_interval: int = 10 # add current model to pool every N updates
    mcts_opponent_ratio: float = 0.3 # fraction of games vs MCTS (0.0 = pure self-play, 1.0 = pure MCTS)
    mcts_opponent_playouts: int = 100 # MCTS playouts per move during training

    # --- Rewards ---
    reward_win: float = 1.0
    reward_loss: float = -1.0
    reward_draw: float = 0.0
    reward_step: float = -0.001  # small penalty per step to encourage faster play
    reward_capture: float = 0.05 # reward for each captured opponent piece
    max_game_steps: int = 300    # max steps before forced draw

    # --- Logging & checkpoints ---
    log_interval: int = 10       # log every N updates
    save_interval: int = 100     # save model every N updates
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    # --- Device ---
    use_gpu: bool = True
