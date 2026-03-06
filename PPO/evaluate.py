"""
Evaluate a trained PPO model against the Pure MCTS player from deepdraughts.

Usage:
    cd PPO
    python evaluate.py --checkpoint checkpoints/ppo_checkers_1100.pt --games 50 --mcts-playouts 1000

The script plays N games (PPO as WHITE, then PPO as BLACK) and prints
win/loss/draw statistics.
"""

import sys
import os
import argparse
import time
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from config import PPOConfig
from model import ActorCritic
from deepdraughts.env.py_env.game import Game
from deepdraughts.env.py_env.env_utils import (
    WHITE, BLACK, RUSSIAN_RULE,
    GAME_CONTINUE, GAME_WHITE_WIN, GAME_BLACK_WIN, GAME_DRAW,
    game_is_over, game_is_drawn, game_winner, game_status_to_str,
    state2vec, action2id, N_ACTION_64,
)
from deepdraughts.mcts_pure import MCTSPlayer
from env_wrapper import canonical_observation


# ── PPO move selection ─────────────────────────────────────────────

def ppo_select_move(model, game, device):
    """Use trained PPO model to pick a move (greedy argmax)."""
    (vb, vs), mask, legal_map = canonical_observation(game)

    vb_t = torch.tensor(vb, device=device, dtype=torch.float32).unsqueeze(0)
    vs_t = torch.tensor(vs, device=device, dtype=torch.float32).unsqueeze(0)
    am_t = torch.tensor(mask, device=device, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits, _ = model(vb_t, vs_t, am_t)

    action_id = logits.argmax(dim=-1).item()
    return legal_map[action_id]


# ── Single game ────────────────────────────────────────────────────

def play_one_game(model, device, mcts_player, ppo_color, max_steps=300):
    """
    Play one game: PPO vs MCTS.

    Args:
        ppo_color: WHITE or BLACK — which side PPO plays.

    Returns:
        winner: WHITE(1), BLACK(-1), or 0(draw)
        n_steps: number of moves played
    """
    game = Game(rule=RUSSIAN_RULE)
    mcts_player.reset()

    for step in range(1, max_steps + 1):
        if game.current_player == ppo_color:
            move = ppo_select_move(model, game, device)
        else:
            move, _ = mcts_player.get_action(game)

        game_status = game.do_move(move)

        if game_is_over(game_status):
            winner = game_winner(game_status)
            return winner, step

    # Hit step limit → draw
    return 0, max_steps


# ── Main evaluation ───────────────────────────────────────────────

def evaluate(checkpoint_path, n_games, mcts_playouts, use_gpu=False, max_steps=300):
    cfg = PPOConfig()
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Load PPO model
    model = ActorCritic(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    update_count = checkpoint.get("update_count", "?")
    print(f"Loaded PPO model: {checkpoint_path} (update {update_count})")

    # Create MCTS player
    mcts_player = MCTSPlayer(c_puct=5, n_playout=mcts_playouts)
    print(f"MCTS opponent: {mcts_playouts} playouts")
    print(f"Games per side: {n_games} (total: {n_games * 2})")
    print("-" * 60)

    # Stats
    results = {
        "ppo_wins": 0,
        "mcts_wins": 0,
        "draws": 0,
        "ppo_wins_as_white": 0,
        "ppo_wins_as_black": 0,
        "mcts_wins_as_white": 0,
        "mcts_wins_as_black": 0,
        "draws_as_white": 0,
        "draws_as_black": 0,
        "total_steps": 0,
        "game_lengths": [],
    }

    total_games = n_games * 2
    game_num = 0
    start_time = time.time()

    # --- PPO plays as WHITE ---
    print("\n[Phase 1] PPO as WHITE vs MCTS as BLACK")
    for i in range(1, n_games + 1):
        game_num += 1
        winner, steps = play_one_game(model, device, mcts_player, WHITE, max_steps)
        results["total_steps"] += steps
        results["game_lengths"].append(steps)

        if winner == WHITE:
            results["ppo_wins"] += 1
            results["ppo_wins_as_white"] += 1
            result_str = "PPO wins"
        elif winner == BLACK:
            results["mcts_wins"] += 1
            results["mcts_wins_as_black"] += 1
            result_str = "MCTS wins"
        else:
            results["draws"] += 1
            results["draws_as_white"] += 1
            result_str = "Draw"

        elapsed = time.time() - start_time
        print(f"  Game {game_num}/{total_games}: {result_str} ({steps} moves) [{elapsed:.0f}s]")

    # --- PPO plays as BLACK ---
    print("\n[Phase 2] PPO as BLACK vs MCTS as WHITE")
    for i in range(1, n_games + 1):
        game_num += 1
        winner, steps = play_one_game(model, device, mcts_player, BLACK, max_steps)
        results["total_steps"] += steps
        results["game_lengths"].append(steps)

        if winner == BLACK:
            results["ppo_wins"] += 1
            results["ppo_wins_as_black"] += 1
            result_str = "PPO wins"
        elif winner == WHITE:
            results["mcts_wins"] += 1
            results["mcts_wins_as_white"] += 1
            result_str = "MCTS wins"
        else:
            results["draws"] += 1
            results["draws_as_black"] += 1
            result_str = "Draw"

        elapsed = time.time() - start_time
        print(f"  Game {game_num}/{total_games}: {result_str} ({steps} moves) [{elapsed:.0f}s]")

    # --- Summary ---
    total_time = time.time() - start_time
    avg_len = results["total_steps"] / total_games

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total games:        {total_games}")
    print(f"MCTS playouts:      {mcts_playouts}")
    print(f"PPO model update:   {update_count}")
    print()
    print(f"PPO wins:           {results['ppo_wins']}/{total_games}  ({results['ppo_wins']/total_games:.1%})")
    print(f"  as WHITE:         {results['ppo_wins_as_white']}/{n_games}")
    print(f"  as BLACK:         {results['ppo_wins_as_black']}/{n_games}")
    print(f"MCTS wins:          {results['mcts_wins']}/{total_games}  ({results['mcts_wins']/total_games:.1%})")
    print(f"  as WHITE:         {results['mcts_wins_as_white']}/{n_games}")
    print(f"  as BLACK:         {results['mcts_wins_as_black']}/{n_games}")
    print(f"Draws:              {results['draws']}/{total_games}  ({results['draws']/total_games:.1%})")
    print()
    print(f"Avg game length:    {avg_len:.1f} moves")
    print(f"Total time:         {total_time:.1f}s ({total_time/total_games:.1f}s/game)")
    print("=" * 60)


def find_latest_checkpoint(checkpoint_dir="checkpoints"):
    """Find the most recent last_check_*.pt file."""
    pattern = os.path.join(checkpoint_dir, "last_check_*.pt")
    files = glob.glob(pattern)
    if not files:
        return None
    # Extract step count from filename and pick the highest
    return max(files, key=os.path.getmtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO vs Pure MCTS")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to PPO .pt checkpoint (default: latest last_check_*.pt)")
    parser.add_argument("--games", type=int, default=25, help="Number of games PER SIDE (total = 2x this)")
    parser.add_argument("--mcts-playouts", type=int, default=1000, help="MCTS playouts per move (higher = stronger)")
    parser.add_argument("--max-steps", type=int, default=300, help="Max moves per game")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    args = parser.parse_args()

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = find_latest_checkpoint()
        if checkpoint is None:
            print("Error: No checkpoint found in checkpoints/. Use --checkpoint to specify one.")
            sys.exit(1)
        print(f"Auto-detected latest checkpoint: {checkpoint}")

    evaluate(checkpoint, args.games, args.mcts_playouts, args.gpu, args.max_steps)
