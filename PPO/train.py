"""
PPO Self-Play Training Loop for Checkers.

The agent plays against itself: at each step the same network picks
moves for the current player.  Transitions are collected into a rollout
buffer and then PPO updates the policy.

"""
import sys
import os
import time
import copy
import random
import argparse
import glob

# Make sure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import PPOConfig
from model import ActorCritic
from env_wrapper import SelfPlayEnv
from ppo_algo import PPO
from utils import Timer, ensure_dir, set_seed

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
    from env_wrapper import _FLIPPED_ACTION
    action = action2id(move)
    # The wrapper exposes flipped action IDs when Black is playing;
    # MCTS returns the original move, so we must flip the ID to match.
    if env.env._is_flipped:
        action = _FLIPPED_ACTION.get(action, action)
    _, _, done, info = env.step(action)
    return done, info


def _record_result(winner, learner_color, info, counters):
    """Helper: update stats and learner win/loss/draw counters."""
    counters["games"] += 1
    counters["total_steps"] += info["step_count"]
    
    # Track by color
    if winner == 1:  # WHITE
        counters["white_wins"] += 1
    elif winner == -1:  # BLACK
        counters["black_wins"] += 1
    else:
        counters["draws"] += 1
    
    # Track learner performance
    if winner == learner_color:
        counters["learner_wins"] += 1
    elif winner != 0:
        counters["learner_losses"] += 1
    else:
        counters["learner_draws"] += 1


def collect_rollout(ppo: PPO, envs: list, config: PPOConfig,
                    opponent_model, opponent_pool: OpponentPool,
                    mcts_player: MCTSPlayer | None = None):
    """
    Collect a rollout from N parallel self-play environments.

    Each env fills a contiguous block in the buffer so that GAE
    bootstrapping stays correct (consecutive entries belong to the
    same env).
    """
    buffer = ppo.buffer
    buffer.reset()
    ppo.model.eval()

    n_envs = len(envs)
    steps_per_env = config.rollout_steps // n_envs

    counters = {
        "games": 0, "learner_wins": 0, "learner_losses": 0, "learner_draws": 0,
        "mcts_games": 0, "pool_games": 0,
        "white_wins": 0, "black_wins": 0, "draws": 0, "total_steps": 0,
    }
    env_blocks = []  # (start, end, last_value, last_done) per env

    for env_idx in range(n_envs):
        env = envs[env_idx]
        block_start = buffer.pos
        stored = 0
        game_active = False
        use_mcts = False
        obs = None
        mask = None
        learner_color = WHITE

        while stored < steps_per_env:
            # ── Start a new game if needed ────────────────────────
            if not game_active:
                obs, mask = env.reset()
                learner_color = random.choice([WHITE, BLACK])

                use_mcts = (mcts_player is not None
                            and random.random() < config.mcts_opponent_ratio)

                if use_mcts:
                    mcts_player.reset()
                    counters["mcts_games"] += 1
                else:
                    snap = opponent_pool.sample()
                    opponent_model.load_state_dict(
                        snap if snap is not None else ppo.model.state_dict()
                    )
                    opponent_model.eval()
                    counters["pool_games"] += 1

                opp_ended = False
                while env.get_current_player() != learner_color:
                    if use_mcts:
                        done, info = _mcts_opponent_move(env, mcts_player)
                    else:
                        done, info = _opponent_move(env, opponent_model, ppo.device)
                    if done:
                        gs = info["game_status"]
                        w = game_winner(gs) if game_is_over(gs) else 0
                        _record_result(w, learner_color, info, counters)
                        opp_ended = True
                        break
                if opp_ended:
                    continue

                obs = env.env._get_obs()
                mask = env.env.get_action_mask()
                game_active = True

            # ── Learner picks a move ──────────────────────────────
            vb, vs = obs
            action, log_prob, value = ppo.select_action(vb, vs, mask)
            _, reward_l, done_l, info_l = env.step(action)

            if done_l:
                buffer.add(vb, vs, mask, action, log_prob, reward_l, value, True)
                stored += 1
                gs = info_l["game_status"]
                w = game_winner(gs) if game_is_over(gs) else 0
                _record_result(w, learner_color, info_l, counters)
                game_active = False
                continue

            if env.get_current_player() == learner_color:
                buffer.add(vb, vs, mask, action, log_prob, reward_l, value, False)
                stored += 1
                obs = env.env._get_obs()
                mask = env.env.get_action_mask()
                continue

            # ── Opponent's turn(s) ────────────────────────────────
            opp_ended_game = False
            while env.get_current_player() != learner_color:
                if use_mcts:
                    done_o, info_o = _mcts_opponent_move(env, mcts_player)
                else:
                    done_o, info_o = _opponent_move(env, opponent_model, ppo.device)
                if done_o:
                    gs = info_o["game_status"]
                    w = game_winner(gs) if game_is_over(gs) else 0
                    if w == learner_color:
                        terminal_r = config.reward_win
                    elif w != 0:
                        terminal_r = config.reward_loss
                    else:
                        terminal_r = config.reward_draw
                    buffer.add(vb, vs, mask, action, log_prob,
                               reward_l + terminal_r, value, True)
                    stored += 1
                    _record_result(w, learner_color, info_o, counters)
                    game_active = False
                    opp_ended_game = True
                    break

            if not opp_ended_game:
                buffer.add(vb, vs, mask, action, log_prob,
                           reward_l, value, False)
                stored += 1
                obs = env.env._get_obs()
                mask = env.env.get_action_mask()

        # ── Bootstrap this env's block ────────────────────────────
        block_end = buffer.pos
        if game_active:
            vb_last, vs_last = obs
            last_val = ppo.estimate_value(vb_last, vs_last, mask)
            env_blocks.append((block_start, block_end, last_val, False))
        else:
            env_blocks.append((block_start, block_end, 0.0, True))

    # ── Compute GAE per env block ─────────────────────────────────
    for start, end, last_val, last_done in env_blocks:
        buffer.compute_gae_range(last_val, last_done, start, end)

    return counters


def evaluate_vs_mcts(model, device, config, n_games=10, mcts_playouts=100):
    """
    Play n_games against Pure MCTS and return PPO win rate.
    Half the games are played as WHITE, half as BLACK.
    """
    from evaluate import play_one_game
    mcts_player = MCTSPlayer(c_puct=5, n_playout=mcts_playouts)
    ppo_wins = 0
    half = n_games // 2

    for _ in range(half):
        mcts_player.reset()
        winner, _ = play_one_game(model, device, mcts_player, WHITE, config.max_game_steps)
        if winner == WHITE:
            ppo_wins += 1

    for _ in range(n_games - half):
        mcts_player.reset()
        winner, _ = play_one_game(model, device, mcts_player, BLACK, config.max_game_steps)
        if winner == BLACK:
            ppo_wins += 1

    return ppo_wins / max(n_games, 1)


def train(config: PPOConfig, resume_path: str | None = None):
    """Main training loop."""
    set_seed(42)
    ensure_dir(config.checkpoint_dir)
    ensure_dir(config.log_dir)

    writer = SummaryWriter(log_dir=config.log_dir)
    timer = Timer()

    # Text log file — appends so resumed runs continue the same file
    log_file_path = os.path.join(config.checkpoint_dir, "training_log.txt")
    is_fresh_start = resume_path is None
    log_file = open(log_file_path, "a", encoding="utf-8")

    def log(msg: str, file_only_on_resume: bool = False):
        """Print to terminal and append to log file.
        If file_only_on_resume=True and we're resuming, skip writing to log file."""
        print(msg)
        if not (file_only_on_resume and not is_fresh_start):
            log_file.write(msg + "\n")
            log_file.flush()

    ppo = PPO(config)
    envs = [SelfPlayEnv(config) for _ in range(config.n_envs)]

    # Opponent pool for diverse training
    opponent_model = ActorCritic(config).to(ppo.device)
    opponent_pool = OpponentPool(max_size=config.opponent_pool_size)
    opponent_pool.add(ppo.model)

    # MCTS opponent (None if ratio == 0)
    mcts_player = None
    if config.mcts_opponent_ratio > 0:
        mcts_player = MCTSPlayer(c_puct=5, n_playout=config.mcts_opponent_playouts)
        log(f"MCTS opponent: {config.mcts_opponent_playouts} playouts, "
            f"{config.mcts_opponent_ratio:.0%} of games", file_only_on_resume=True)

    total_steps = 0
    start_update = 0

    # --- Resume from checkpoint ---
    if resume_path is not None:
        total_steps = ppo.load(resume_path)
        start_update = ppo.update_count
        opponent_pool.add(ppo.model)  # add resumed model to pool
        log(f"Resumed from {resume_path}", file_only_on_resume=True)
        log(f"  update_count = {start_update}, total_steps = {total_steps:,}", file_only_on_resume=True)

    update_num = start_update
    n_updates = config.total_timesteps // config.rollout_steps

    log(f"Device: {ppo.device}", file_only_on_resume=True)
    log(f"Total timesteps: {config.total_timesteps:,}", file_only_on_resume=True)
    log(f"Rollout steps: {config.rollout_steps}", file_only_on_resume=True)
    log(f"Number of updates: {n_updates}", file_only_on_resume=True)
    log(f"Model parameters: {sum(p.numel() for p in ppo.model.parameters()):,}", file_only_on_resume=True)
    log("-" * 60, file_only_on_resume=True)

    for update in range(start_update + 1, n_updates + 1):
        # --- Collect rollout ---
        rollout_info = collect_rollout(ppo, envs, config,
                                       opponent_model, opponent_pool,
                                       mcts_player)
        total_steps += config.rollout_steps

        # --- Learning rate & entropy coefficient schedule (linear decay) ---
        progress = update / n_updates
        current_lr = config.lr * max(1.0 - progress, config.lr_min_fraction)
        for param_group in ppo.optimizer.param_groups:
            param_group['lr'] = current_lr
        current_entropy_coef = config.entropy_coef + (config.entropy_coef_end - config.entropy_coef) * progress

        # --- PPO update ---
        losses = ppo.update(entropy_coef=current_entropy_coef)
        update_num += 1

        # --- Add model to opponent pool ---
        if update % config.opponent_pool_interval == 0:
            opponent_pool.add(ppo.model)

        # --- Logging ---
        writer.add_scalar("loss/policy", losses["policy_loss"], total_steps)
        writer.add_scalar("loss/value", losses["value_loss"], total_steps)
        writer.add_scalar("loss/entropy", losses["entropy"], total_steps)
        writer.add_scalar("loss/total", losses["total_loss"], total_steps)
        writer.add_scalar("schedule/lr", current_lr, total_steps)
        writer.add_scalar("schedule/entropy_coef", current_entropy_coef, total_steps)

        ri = rollout_info
        if ri["games"] > 0:
            white_wr = ri["white_wins"] / ri["games"]
            black_wr = ri["black_wins"] / ri["games"]
            draw_r = ri["draws"] / ri["games"]
            avg_len = ri["total_steps"] / ri["games"]
            lwr = ri["learner_wins"] / ri["games"]
            writer.add_scalar("game/white_win_rate", white_wr, total_steps)
            writer.add_scalar("game/black_win_rate", black_wr, total_steps)
            writer.add_scalar("game/draw_rate", draw_r, total_steps)
            writer.add_scalar("game/avg_length", avg_len, total_steps)
            writer.add_scalar("game/learner_winrate", lwr, total_steps)
        writer.add_scalar("pool/size", len(opponent_pool), total_steps)
        if ri["mcts_games"] + ri["pool_games"] > 0:
            writer.add_scalar("pool/mcts_games", ri["mcts_games"], total_steps)
            writer.add_scalar("pool/pool_games", ri["pool_games"], total_steps)

        if update % config.log_interval == 0:
            elapsed = timer.elapsed_str()
            sps = total_steps / timer.elapsed()
            n_games = max(ri["games"], 1)
            lwr = ri["learner_wins"] / n_games
            white_wr = ri["white_wins"] / n_games
            black_wr = ri["black_wins"] / n_games
            draw_r = ri["draws"] / n_games
            avg_len = ri["total_steps"] / n_games
            opp_str = f"Pool:{ri['pool_games']} MCTS:{ri['mcts_games']}"
            game_str = f"Games: {ri['games']} | W: {white_wr:.0%} | B: {black_wr:.0%} | D: {draw_r:.0%} | Len: {avg_len:.0f}"
            log(
                f"Update {update}/{n_updates} | "
                f"Steps: {total_steps:,} | "
                f"SPS: {sps:.0f} | "
                f"LWR: {lwr:.0%} | "
                f"{opp_str} | "
                f"{game_str} | "
                f"PL: {losses['policy_loss']:.4f} | "
                f"VL: {losses['value_loss']:.4f} | "
                f"Ent: {losses['entropy']:.4f} | "
                f"Time: {elapsed}"
            )

        # --- Save last checkpoint (always overwritten) ---
        last_path = os.path.join(config.checkpoint_dir, f"last_check_{total_steps}.pt")
        ppo.save(last_path, total_steps=total_steps)
        # Remove previous "last_check_*" to avoid clutter
        for f in os.listdir(config.checkpoint_dir):
            if f.startswith("last_check_") and f != os.path.basename(last_path):
                os.remove(os.path.join(config.checkpoint_dir, f))

    # Final save
    final_path = os.path.join(config.checkpoint_dir, "ppo_checkers_final.pt")
    ppo.save(final_path, total_steps=total_steps)
    log(f"\nTraining complete. Final model saved to {final_path}")
    log(f"Total time: {timer.elapsed_str()}")
    log_file.close()
    writer.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Checkers Training")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--total-steps", type=int, default=None, help="Total timesteps")
    parser.add_argument("--rollout-steps", type=int, default=None, help="Rollout length")
    parser.add_argument("--batch-size", type=int, default=None, help="Mini-batch size")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    parser.add_argument("--resume", type=str, nargs="?", const="auto", default=None,
                        help="Resume training. Optionally provide a path; without a path, uses latest last_check_*.pt")
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

    # Auto-detect latest checkpoint if --resume given without path
    resume_path = args.resume
    if resume_path == "auto":
        pattern = os.path.join(cfg.checkpoint_dir, "last_check_*.pt")
        files = glob.glob(pattern)
        if files:
            resume_path = max(files, key=os.path.getmtime)
            print(f"Auto-detected latest checkpoint: {resume_path}")
        else:
            print(f"Error: No last_check_*.pt found in {cfg.checkpoint_dir}/")
            sys.exit(1)

    train(cfg, resume_path=resume_path)
