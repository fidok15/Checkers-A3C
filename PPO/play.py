"""
Play a game of Checkers against the trained PPO model.

Usage:
    cd PPO
    python play.py --checkpoint checkpoints/ppo_checkers_200.pt

You play as WHITE (moves first), the AI plays as BLACK.
Use --play-as black to switch sides.
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import numpy as np

from config import PPOConfig
from model import ActorCritic
from deepdraughts.env.py_env.game import Game
from deepdraughts.env.py_env.env_utils import (
    WHITE, BLACK, RUSSIAN_RULE,
    GAME_CONTINUE, GAME_WHITE_WIN, GAME_BLACK_WIN, GAME_DRAW,
    game_is_over, game_winner, game_status_to_str,
    state2vec, action2id, N_ACTION_64,
    pos2coord, coord2pos, CONST_N_SIZE_8, VALID_POS_64,
)
from env_wrapper import canonical_observation


# ── Board display ──────────────────────────────────────────────────

PIECE_SYMBOLS = {
    (WHITE, False): "⛀ ",   # white man
    (WHITE, True):  "⛁ ",   # white king
    (BLACK, False): "⛂ ",   # black man
    (BLACK, True):  "⛃ ",   # black king
}

PIECE_SYMBOLS_ASCII = {
    (WHITE, False): "w ",
    (WHITE, True):  "W ",
    (BLACK, False): "b ",
    (BLACK, True):  "B ",
}


def print_board(game: Game, use_unicode: bool = True):
    """Print the board to the console."""
    symbols = PIECE_SYMBOLS if use_unicode else PIECE_SYMBOLS_ASCII
    n = game.current_board.nsize  # 8

    print()
    print("    A  B  C  D  E  F  G  H")
    print("  ┌" + "──┬" * 7 + "──┐")

    for row in range(n):
        chess_row = n - row  # 8..1 (chess notation)
        pieces_str = ""
        for col in range(n):
            chess_col = col + 1  # 1..8 (A=1, H=8)
            pos = coord2pos(chess_row, chess_col, n)  # internal position
            piece = game.current_board.pieces.get(pos)
            if piece is not None:
                sym = symbols[(piece.player, piece.isking)]
            elif pos in VALID_POS_64:
                sym = "· "  # dark square (no piece)
            else:
                sym = "  "  # light square
            pieces_str += sym + "│"

        print(f"{chess_row} │{pieces_str}")
        if row < n - 1:
            print("  ├" + "──┼" * 7 + "──┤")

    print("  └" + "──┴" * 7 + "──┘")
    print()


# ── AI move selection ──────────────────────────────────────────────

def ai_select_move(model, game, device, ai_color):
    """Use the trained model to pick the best move (greedy)."""
    (vb, vs), mask, legal_map = canonical_observation(game)

    vb_t = torch.tensor(vb, device=device, dtype=torch.float32).unsqueeze(0)
    vs_t = torch.tensor(vs, device=device, dtype=torch.float32).unsqueeze(0)
    am_t = torch.tensor(mask, device=device, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits, value = model(vb_t, vs_t, am_t)

    action_id = logits.argmax(dim=-1).item()
    move = legal_map[action_id]

    # Get confidence
    probs = torch.softmax(logits, dim=-1).squeeze()
    confidence = probs[action_id].item()
    val = value.item()

    return move, confidence, val


# ── Human move selection ───────────────────────────────────────────

def format_move(move) -> str:
    """Format a move for display."""
    pos_from, pos_to = move.pos
    r1, c1 = pos2coord(pos_from, CONST_N_SIZE_8)  # default origin="left_lower"
    r2, c2 = pos2coord(pos_to, CONST_N_SIZE_8)
    col_letters = "ABCDEFGH"
    from_str = f"{col_letters[c1-1]}{r1}"
    to_str = f"{col_letters[c2-1]}{r2}"
    sep = "x" if move.take_piece else "-"
    return f"{from_str}{sep}{to_str}"


def human_select_move(game: Game) -> object:
    """Let the human choose from available moves."""
    moves = game.get_all_available_moves()

    print("Available moves:")
    for i, m in enumerate(moves):
        print(f"  [{i}] {format_move(m)}")

    while True:
        try:
            choice = input("Your move (number): ").strip()
            idx = int(choice)
            if 0 <= idx < len(moves):
                return moves[idx]
            print(f"Enter a number between 0 and {len(moves)-1}")
        except ValueError:
            print("Enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print("\nQuitting.")
            sys.exit(0)


# ── Main game loop ────────────────────────────────────────────────

def play(checkpoint_path: str, play_as: str = "white", use_gpu: bool = False):
    cfg = PPOConfig()
    device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")

    # Load model
    model = ActorCritic(cfg).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    update_count = checkpoint.get("update_count", "?")
    print(f"Loaded model from {checkpoint_path} (update {update_count})")

    human_color = WHITE if play_as == "white" else BLACK
    ai_color = BLACK if human_color == WHITE else WHITE
    color_name = {WHITE: "WHITE ⛀", BLACK: "BLACK ⛂"}

    print(f"You play as {color_name[human_color]}")
    print(f"AI plays as {color_name[ai_color]}")
    print()

    game = Game(rule=RUSSIAN_RULE)
    step = 0

    while True:
        print_board(game)
        current = "WHITE" if game.current_player == WHITE else "BLACK"
        print(f"Turn {step+1} — {current} to move")

        if game.current_player == human_color:
            move = human_select_move(game)
            print(f"You play: {format_move(move)}")
        else:
            move, confidence, value = ai_select_move(model, game, device, ai_color)
            print(f"AI plays: {format_move(move)}  (confidence: {confidence:.1%}, eval: {value:.3f})")

        game_status = game.do_move(move)
        step += 1

        if game_is_over(game_status):
            print_board(game)
            print(game_status_to_str(game_status))
            winner = game_winner(game_status)
            if winner == human_color:
                print("🎉 You win!")
            elif winner == 0:
                print("🤝 Draw!")
            else:
                print("🤖 AI wins!")
            break

    print(f"\nGame lasted {step} moves.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play Checkers vs trained PPO AI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--play-as", type=str, default="white", choices=["white", "black"],
                        help="Play as white or black (default: white)")
    parser.add_argument("--gpu", action="store_true", help="Use GPU")
    args = parser.parse_args()

    play(args.checkpoint, args.play_as, args.gpu)
