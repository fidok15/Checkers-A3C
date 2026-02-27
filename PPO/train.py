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
import copy
import random
import argparse

# Make sure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import PPOConfig
from model import ActorCritic
from env_wrapper import SelfPlayEnv
from ppo_algo import PPO
from utils import GameStats, Timer, ensure_dir, set_seed

from deepdraughts.env.py_env.env_utils import (
    WHITE, BLACK,
    GAME_WHITE_WIN, GAME_BLACK_WIN, GAME_DRAW,
    game_is_over, game_winner,
)
from deepdraughts.mcts_pure import MCTSPlayer


# ── Opponent Pool ──────────────────────────────────────────────────

class OpponentPool:
    """Stores past model snapshots for diverse training opponents."""

    def __init__(self, max_size: int = 20):
        self.snapshots: list[dict] = []
        self.max_size = max_size

    def add(self, model):
        """Save a deep copy of the model's current weights."""
        self.snapshots.append(copy.deepcopy(model.state_dict()))
        if len(self.snapshots) > self.max_size:
            self.snapshots.pop(0)

    def sample(self) -> dict | None:
        """Return a random past snapshot, or None if empty."""
        return random.choice(self.snapshots) if self.snapshots else None

    def __len__(self):
        return len(self.snapshots)


@torch.no_grad()
def _opponent_move(env: SelfPlayEnv, model, device):
    """Have the opponent model make one move. Returns (done, info)."""
    vb, vs = env.env._get_obs()
    mask = env.env.get_action_mask()

    vb_t = torch.tensor(vb, device=device, dtype=torch.float32).unsqueeze(0)
    vs_t = torch.tensor(vs, device=device, dtype=torch.float32).unsqueeze(0)
    am_t = torch.tensor(mask, device=device, dtype=torch.float32).unsqueeze(0)

    logits, _ = model(vb_t, vs_t, am_t)
    dist = torch.distributions.Categorical(logits=logits)
    action = dist.sample().item()

    _, _, done, info = env.step(action)
    return done, info


def _mcts_opponent_move(env: SelfPlayEnv, mcts_player: MCTSPlayer):
    """Have the MCTS opponent make one move. Returns (done, info)."""
    move, _ = mcts_player.get_action(env.env.game)
    from deepdraughts.env.py_env.env_utils import action2id
    action = action2id(move)
    _, _, done, info = env.step(action)
    return done, info


def _record_result(winner, learner_color, stats, info, counters):
    """Helper: update stats and learner win/loss/draw counters."""
    stats.record(winner, info["step_count"])
    counters["games"] += 1
    if winner == learner_color:
        counters["learner_wins"] += 1
    elif winner != 0:
        counters["learner_losses"] += 1
    else:
        counters["learner_draws"] += 1


def collect_rollout(ppo: PPO, env: SelfPlayEnv, config: PPOConfig,
                    stats: GameStats, opponent_model, opponent_pool: OpponentPool,
                    mcts_player: MCTSPlayer | None = None):
    """
    Collect a rollout with an opponent sampled from the pool or MCTS.

    With probability config.mcts_opponent_ratio, the opponent is MCTS.
    Otherwise, a random past snapshot from the pool is used.
    """
    buffer = ppo.buffer
    buffer.reset()
    ppo.model.eval()

    stored = 0
    counters = {"games": 0, "learner_wins": 0, "learner_losses": 0,
                "learner_draws": 0, "mcts_games": 0, "pool_games": 0}
    game_active = False
    use_mcts = False

    while stored < config.rollout_steps:
        # ── Start a new game if needed ────────────────────────────
        if not game_active:
            obs, mask = env.reset()
            learner_color = random.choice([WHITE, BLACK])

            # Decide opponent type for this game
            use_mcts = (mcts_player is not None
                        and random.random() < config.mcts_opponent_ratio)

            if use_mcts:
                mcts_player.reset()
                counters["mcts_games"] += 1
            else:
                # Load random opponent from pool
                snap = opponent_pool.sample()
                opponent_model.load_state_dict(
                    snap if snap is not None else ppo.model.state_dict()
                )
                opponent_model.eval()
                counters["pool_games"] += 1

            # Let opponent go first when learner is not the starting side
            opp_ended = False
            while env.get_current_player() != learner_color:
                if use_mcts:
                    done, info = _mcts_opponent_move(env, mcts_player)
                else:
                    done, info = _opponent_move(env, opponent_model, ppo.device)
                if done:
                    gs = info["game_status"]
                    w = game_winner(gs) if game_is_over(gs) else 0
                    _record_result(w, learner_color, stats, info, counters)
                    opp_ended = True
                    break
            if opp_ended:
                continue  # start a fresh game

            obs = env.env._get_obs()
            mask = env.env.get_action_mask()
            game_active = True

        # ── Learner picks a move ──────────────────────────────────
        vb, vs = obs
        action, log_prob, value = ppo.select_action(vb, vs, mask)
        _, reward_l, done_l, info_l = env.step(action)

        if done_l:
            # Learner's move ended the game
            buffer.add(vb, vs, mask, action, log_prob, reward_l, value, True)
            stored += 1
            gs = info_l["game_status"]
            w = game_winner(gs) if game_is_over(gs) else 0
            _record_result(w, learner_color, stats, info_l, counters)
            game_active = False
            continue

        # Still learner's turn?  (chain capture continuation)
        if env.get_current_player() == learner_color:
            buffer.add(vb, vs, mask, action, log_prob, reward_l, value, False)
            stored += 1
            obs = env.env._get_obs()
            mask = env.env.get_action_mask()
            continue

        # ── Opponent's turn(s) ────────────────────────────────────
        opp_ended_game = False
        while env.get_current_player() != learner_color:
            if use_mcts:
                done_o, info_o = _mcts_opponent_move(env, mcts_player)
            else:
                done_o, info_o = _opponent_move(env, opponent_model, ppo.device)
            if done_o:
                gs = info_o["game_status"]
                w = game_winner(gs) if game_is_over(gs) else 0
                # Terminal reward from learner's perspective
                if w == learner_color:
                    terminal_r = config.reward_win
                elif w != 0:
                    terminal_r = config.reward_loss
                else:
                    terminal_r = config.reward_draw
                buffer.add(vb, vs, mask, action, log_prob,
                           reward_l + terminal_r, value, True)
                stored += 1
                _record_result(w, learner_color, stats, info_o, counters)
                game_active = False
                opp_ended_game = True
                break

        if not opp_ended_game:
            # Opponent didn't end the game — store non-terminal transition
            buffer.add(vb, vs, mask, action, log_prob, reward_l, value, False)
            stored += 1
            obs = env.env._get_obs()
            mask = env.env.get_action_mask()

    # ── Bootstrap last incomplete episode ─────────────────────────
    if game_active:
        vb, vs = obs
        last_val = ppo.estimate_value(vb, vs, mask)
        buffer.compute_gae(last_val, False)
    else:
        buffer.compute_gae(0.0, True)

    return counters


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

    # Opponent pool for diverse training
    opponent_model = ActorCritic(config).to(ppo.device)
    opponent_pool = OpponentPool(max_size=config.opponent_pool_size)
    opponent_pool.add(ppo.model)

    # MCTS opponent (None if ratio == 0)
    mcts_player = None
    if config.mcts_opponent_ratio > 0:
        mcts_player = MCTSPlayer(c_puct=5, n_playout=config.mcts_opponent_playouts)
        print(f"MCTS opponent: {config.mcts_opponent_playouts} playouts, "
              f"{config.mcts_opponent_ratio:.0%} of games")

    total_steps = 0
    start_update = 0

    # --- Resume from checkpoint ---
    if resume_path is not None:
        total_steps = ppo.load(resume_path)
        start_update = ppo.update_count
        opponent_pool.add(ppo.model)  # add resumed model to pool
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
        rollout_info = collect_rollout(ppo, env, config, stats,
                                       opponent_model, opponent_pool,
                                       mcts_player)
        total_steps += config.rollout_steps

        # --- PPO update ---
        losses = ppo.update()
        update_num += 1

        # --- Add model to opponent pool ---
        if update % config.opponent_pool_interval == 0:
            opponent_pool.add(ppo.model)

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

        ri = rollout_info
        if ri["games"] > 0:
            lwr = ri["learner_wins"] / ri["games"]
            writer.add_scalar("game/learner_winrate", lwr, total_steps)
        writer.add_scalar("pool/size", len(opponent_pool), total_steps)
        if ri["mcts_games"] + ri["pool_games"] > 0:
            writer.add_scalar("pool/mcts_games", ri["mcts_games"], total_steps)
            writer.add_scalar("pool/pool_games", ri["pool_games"], total_steps)

        if update % config.log_interval == 0:
            elapsed = timer.elapsed_str()
            sps = total_steps / timer.elapsed()
            lwr = ri["learner_wins"] / max(ri["games"], 1)
            opp_str = f"Pool:{ri['pool_games']} MCTS:{ri['mcts_games']}"
            print(
                f"Update {update}/{n_updates} | "
                f"Steps: {total_steps:,} | "
                f"SPS: {sps:.0f} | "
                f"LWR: {lwr:.0%} | "
                f"{opp_str} | "
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
    parser.add_argument("--mcts-ratio", type=float, default=None, help="Fraction of games vs MCTS (0.0-1.0)")
    parser.add_argument("--mcts-playouts", type=int, default=None, help="MCTS playouts per move during training")
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
    if args.mcts_ratio is not None:
        cfg.mcts_opponent_ratio = args.mcts_ratio
    if args.mcts_playouts is not None:
        cfg.mcts_opponent_playouts = args.mcts_playouts
    if args.checkpoint_dir is not None:
        cfg.checkpoint_dir = args.checkpoint_dir
    if args.log_dir is not None:
        cfg.log_dir = args.log_dir

    train(cfg, resume_path=args.resume)
