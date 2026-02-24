"""
PPO Self-Play Training Loop for Checkers.

The agent plays against itself: at each step the same network picks
moves for the current player.  Transitions are collected into a rollout
buffer and then PPO updates the policy.

Usage:
    cd PPO
    python train.py
"""

import sys
import os
import time
import argparse

# Make sure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import PPOConfig
from env_wrapper import SelfPlayEnv
from ppo_algo import PPO
from utils import GameStats, Timer, ensure_dir, set_seed

from deepdraughts.env.py_env.env_utils import (
    WHITE, BLACK,
    GAME_WHITE_WIN, GAME_BLACK_WIN, GAME_DRAW,
    game_is_over, game_winner,
)


def collect_rollout(ppo: PPO, env: SelfPlayEnv, config: PPOConfig, stats: GameStats):
    """
    Play self-play games and fill the rollout buffer.

    Each step:
      1. Observe from current player's (canonical) perspective.
      2. Agent picks an action.
      3. Environment executes the action, returns reward for the mover.
      4. Store transition.
      5. If game over → reset, record result.
    """
    obs, mask = env.reset()
    vec_board, vec_state = obs
    games_this_rollout = 0

    for step in range(config.rollout_steps):
        # Select action
        action, log_prob, value = ppo.select_action(vec_board, vec_state, mask)

        # Step the environment
        next_obs, reward, done, info = env.step(action)
        next_board, next_state = next_obs
        next_mask = env.env.get_action_mask() if not done else np.zeros(config.n_actions, dtype=np.float32)

        # Store transition
        ppo.buffer.add(
            vec_board, vec_state, mask,
            action, log_prob, reward, value, done,
        )

        if done:
            # Record game result
            gs = info["game_status"]
            winner = game_winner(gs) if game_is_over(gs) else 0
            stats.record(winner, info["step_count"])
            games_this_rollout += 1

            # Reset
            obs, mask = env.reset()
            vec_board, vec_state = obs
        else:
            vec_board, vec_state = next_board, next_state
            mask = next_mask

    # Bootstrap value for the last state
    last_value = ppo.estimate_value(vec_board, vec_state, mask)
    last_done = False  # we stopped mid-game
    ppo.buffer.compute_gae(last_value, last_done)

    return games_this_rollout


def evaluate(ppo: PPO, config: PPOConfig, n_games: int = 20) -> dict:
    """
    Play n_games using greedy action selection (argmax) and report stats.
    """
    env = SelfPlayEnv(config)
    wins_w, wins_b, draws = 0, 0, 0
    total_steps = 0

    for _ in range(n_games):
        obs, mask = env.reset()
        done = False
        steps = 0

        while not done and steps < config.max_game_steps:
            vb, vs = obs
            vb_t = torch.tensor(vb, device=ppo.device).unsqueeze(0)
            vs_t = torch.tensor(vs, device=ppo.device).unsqueeze(0)
            am_t = torch.tensor(mask, device=ppo.device).unsqueeze(0)

            with torch.no_grad():
                logits, _ = ppo.model(vb_t, vs_t, am_t)

            action = logits.argmax(dim=-1).item()
            obs, reward, done, info = env.step(action)
            mask = env.env.get_action_mask() if not done else np.zeros(config.n_actions, dtype=np.float32)
            steps += 1

        gs = info.get("game_status", GAME_DRAW)
        w = game_winner(gs) if game_is_over(gs) else 0
        if w == 1:
            wins_w += 1
        elif w == -1:
            wins_b += 1
        else:
            draws += 1
        total_steps += steps

    return {
        "eval_white_win": wins_w / n_games,
        "eval_black_win": wins_b / n_games,
        "eval_draw": draws / n_games,
        "eval_avg_length": total_steps / n_games,
    }


def train(config: PPOConfig, resume_path: str | None = None):
    """Main training loop."""
    set_seed(42)
    ensure_dir(config.checkpoint_dir)
    ensure_dir(config.log_dir)

    writer = SummaryWriter(log_dir=config.log_dir)
    timer = Timer()

    ppo = PPO(config)
    env = SelfPlayEnv(config)
    stats = GameStats(window=100)

    total_steps = 0
    start_update = 0

    # --- Resume from checkpoint ---
    if resume_path is not None:
        total_steps = ppo.load(resume_path)
        start_update = ppo.update_count
        print(f"Resumed from {resume_path}")
        print(f"  update_count = {start_update}, total_steps = {total_steps:,}")

    update_num = start_update
    n_updates = config.total_timesteps // config.rollout_steps

    print(f"Device: {ppo.device}")
    print(f"Total timesteps: {config.total_timesteps:,}")
    print(f"Rollout steps: {config.rollout_steps}")
    print(f"Number of updates: {n_updates}")
    print(f"Model parameters: {sum(p.numel() for p in ppo.model.parameters()):,}")
    print("-" * 60)

    for update in range(start_update + 1, n_updates + 1):
        # --- Collect rollout ---
        ppo.model.eval()
        games_played = collect_rollout(ppo, env, config, stats)
        total_steps += config.rollout_steps

        # --- PPO update ---
        losses = ppo.update()
        update_num += 1

        # --- Logging ---
        writer.add_scalar("loss/policy", losses["policy_loss"], total_steps)
        writer.add_scalar("loss/value", losses["value_loss"], total_steps)
        writer.add_scalar("loss/entropy", losses["entropy"], total_steps)
        writer.add_scalar("loss/total", losses["total_loss"], total_steps)

        if stats.n_games > 0:
            writer.add_scalar("game/white_win_rate", stats.white_win_rate, total_steps)
            writer.add_scalar("game/black_win_rate", stats.black_win_rate, total_steps)
            writer.add_scalar("game/draw_rate", stats.draw_rate, total_steps)
            writer.add_scalar("game/avg_length", stats.avg_length, total_steps)

        if update % config.log_interval == 0:
            elapsed = timer.elapsed_str()
            sps = total_steps / timer.elapsed()
            print(
                f"Update {update}/{n_updates} | "
                f"Steps: {total_steps:,} | "
                f"SPS: {sps:.0f} | "
                f"{stats.summary()} | "
                f"PL: {losses['policy_loss']:.4f} | "
                f"VL: {losses['value_loss']:.4f} | "
                f"Ent: {losses['entropy']:.4f} | "
                f"Time: {elapsed}"
            )

        # --- Checkpoint ---
        if update % config.save_interval == 0:
            path = os.path.join(config.checkpoint_dir, f"ppo_checkers_{update}.pt")
            ppo.save(path, total_steps=total_steps)
            print(f"  -> Saved checkpoint: {path}")

        # --- Periodic evaluation ---
        if update % (config.save_interval) == 0:
            ppo.model.eval()
            eval_stats = evaluate(ppo, config, n_games=20)
            for k, v in eval_stats.items():
                writer.add_scalar(f"eval/{k}", v, total_steps)
            print(
                f"  -> Eval: W={eval_stats['eval_white_win']:.0%} "
                f"B={eval_stats['eval_black_win']:.0%} "
                f"D={eval_stats['eval_draw']:.0%} "
                f"Len={eval_stats['eval_avg_length']:.0f}"
            )

    # Final save
    final_path = os.path.join(config.checkpoint_dir, "ppo_checkers_final.pt")
    ppo.save(final_path, total_steps=total_steps)
    print(f"\nTraining complete. Final model saved to {final_path}")
    print(f"Total time: {timer.elapsed_str()}")
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Checkers Training")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--total-steps", type=int, default=None, help="Total timesteps")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Rollout length")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint .pt file to resume training")
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = PPOConfig()
    if args.lr is not None:
        cfg.lr = args.lr
    if args.total_steps is not None:
        cfg.total_timesteps = args.total_steps
    if args.rollout_steps is not None:
        cfg.rollout_steps = args.rollout_steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.no_gpu:
        cfg.use_gpu = False
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir

    train(cfg, resume_path=args.resume)
