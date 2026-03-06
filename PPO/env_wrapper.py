'''
Konwertuje zasady gry do tensorów
- Maskowanie akcji (tylko legalne ruchy)
- Mapowanie między indeksami akcji (0..279) a obiektami Move
- Samo-granie: agent gra zarówno jako WHITE, jak i BLACK
- Kształtowanie nagród

'''
import sys
import os
import numpy as np
import copy

# Add the project root so deepdraughts is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from deepdraughts.env.py_env.game import Game
from deepdraughts.env.py_env.env_utils import (
    WHITE, BLACK, RUSSIAN_RULE,
    GAME_CONTINUE, GAME_WHITE_WIN, GAME_BLACK_WIN, GAME_DRAW,
    game_is_over, game_winner,
    state2vec, action2id, N_ACTION_64, MOVE_MAP_64,
)
from deepdraughts.env.py_env.piece import Piece

from config import PPOConfig


# ── Action flipping for Black perspective ──────────────────────────
def _flip_pos(pos: int) -> int:
    """Flip position 180° on the 8x8 board (maps valid dark squares to valid dark squares)."""
    return 63 - pos


# Build reverse map: action_id -> (from_pos, to_pos)
_ID_TO_MOVE = {aid: move for move, aid in MOVE_MAP_64.items()}

# Build flipped action mapping: original_aid -> flipped_aid
_FLIPPED_ACTION = {}
for orig_aid, (from_pos, to_pos) in _ID_TO_MOVE.items():
    flipped_from = _flip_pos(from_pos)
    flipped_to = _flip_pos(to_pos)
    flipped_move = (flipped_from, flipped_to)
    if flipped_move in MOVE_MAP_64:
        _FLIPPED_ACTION[orig_aid] = MOVE_MAP_64[flipped_move]
    else:
        # Fallback: keep original (shouldn't happen for valid moves)
        _FLIPPED_ACTION[orig_aid] = orig_aid

# Reverse: flipped_aid -> original_aid
_UNFLIP_ACTION = {v: k for k, v in _FLIPPED_ACTION.items()}


def canonical_observation(game):
    """
    Get canonical observation and action mapping for the current player.
    Handles board 180° rotation, channel swapping, and action flipping for Black.
    Used by evaluate.py and play.py for consistent perspective handling.

    Returns:
        (vec_board, vec_state): canonical float32 observation tensors
        mask: action mask ndarray of shape (280,)
        legal_map: dict mapping canonical action_id -> Move object
    """
    vec_board, vec_state = state2vec(game)
    is_black = (game.current_player == BLACK)

    if is_black:
        vec_board = vec_board.copy()
        vec_board = np.concatenate([vec_board[2:4], vec_board[0:2]], axis=0)
        vec_board = vec_board[:, ::-1, ::-1].copy()
        vec_state = vec_state.copy()
        vec_state[0] = 1
        n_chain = len(game.chain_taking_moves)
        for i in range(n_chain):
            vec_state[i + 2] = 62 - vec_state[i + 2]

    moves = game.get_all_available_moves()
    legal_map = {}
    for m in moves:
        orig_aid = action2id(m)
        exposed_aid = _FLIPPED_ACTION.get(orig_aid, orig_aid) if is_black else orig_aid
        legal_map[exposed_aid] = m

    mask = np.zeros(N_ACTION_64, dtype=np.float32)
    for aid in legal_map:
        mask[aid] = 1.0

    return (vec_board.astype(np.float32), vec_state.astype(np.float32)), mask, legal_map


class CheckersEnv:
    """
    Bierzemy obserwacje z `state2vec`:
      - vec_board: np.array of shape (4, 8, 8)
      - vec_state: np.array of shape (19,)

    Akcje to integery w [0, N_ACTION_64)  (280 possible moves).
    W każdym ruchu dostępna jest tylko część możliwych ruchów;  `get_action_mask()`.

    Agent zawsze widzi plansze jakby grał białymi, jeśli gra czarnymi,
    to wrapper odwraca planszę, cechy i AKCJE tak, żeby się zgadzały jakby grał białymi.
    """

    def __init__(self, config: PPOConfig | None = None):
        self.config = config or PPOConfig()
        self.game: Game | None = None
        self.n_actions = N_ACTION_64  # 280
        self.step_count = 0

        # Mapping from action_id -> Move for current legal moves
        # When Black is playing, this maps FLIPPED action_ids to original Moves
        self._legal_map: dict[int, object] = {}
        self._is_flipped: bool = False  # True when current player is Black


    def reset(self):
        """Start a new game.  Returns (obs, action_mask)."""
        self.game = Game(rule=RUSSIAN_RULE)
        self.step_count = 0
        self._refresh_legal_map()
        return self._get_obs(), self.get_action_mask()

    def step(self, action_id: int):
        """
            obs        – next observation (from next player's perspective)
            reward     – reward for the player who just moved
            done       – whether the game ended
            info       – dict with extra data
        """
        move = self._legal_map.get(action_id)
        if move is None:
            raise ValueError(
                f"Action {action_id} is not legal. Legal: {list(self._legal_map.keys())}"
            )

        player_before = self.game.current_player
        opponent = BLACK if player_before == WHITE else WHITE

        # Count opponent pieces BEFORE the move (to detect captures)
        opp_pieces_before = sum(
            1 for p in self.game.current_board.pieces.values()
            if p.player == opponent
        )

        game_status = self.game.do_move(move)
        self.step_count += 1

        # Count opponent pieces AFTER the move
        opp_pieces_after = sum(
            1 for p in self.game.current_board.pieces.values()
            if p.player == opponent
        )
        captured = opp_pieces_before - opp_pieces_after

        done = game_is_over(game_status) or self.step_count >= self.config.max_game_steps
        reward = self._compute_reward(game_status, player_before, done)

        # Reward shaping: bonus for capturing opponent pieces
        if captured > 0 and not game_is_over(game_status):
            reward += captured * self.config.reward_capture

        if not done:
            self._refresh_legal_map()
            # If the current player has no legal moves the game is effectively over
            if len(self._legal_map) == 0:
                done = True
                reward = self.config.reward_win  # mover wins — opponent has no legal moves

        obs = self._get_obs()
        mask = self.get_action_mask()

        info = {
            "game_status": game_status,
            "step_count": self.step_count,
            "current_player": self.game.current_player,
            "player_before": player_before,
            "captured": captured,
        }
        return obs, reward, done, info

    def get_action_mask(self) -> np.ndarray:
        """Binary mask of shape (280,). 1 = legal, 0 = illegal."""
        mask = np.zeros(self.n_actions, dtype=np.float32)
        for aid in self._legal_map:
            mask[aid] = 1.0
        return mask

    def get_current_player(self) -> int:
        return self.game.current_player

    
    def _refresh_legal_map(self):
        """
        Rebuild the mapping action_id -> Move for current legal moves.
        When Black is playing, we map FLIPPED action_ids to original Moves,
        so the agent sees consistent action space with the flipped board.
        """
        self._legal_map = {}
        self._is_flipped = (self.game.current_player == BLACK)
        moves = self.game.get_all_available_moves()
        for m in moves:
            orig_aid = action2id(m)
            # If Black is playing, use flipped action ID so it matches flipped board
            exposed_aid = _FLIPPED_ACTION.get(orig_aid, orig_aid) if self._is_flipped else orig_aid
            self._legal_map[exposed_aid] = m

    def _get_obs(self):
        """
        Return (vec_board, vec_state) with a canonical perspective:
        the current player is always treated as WHITE.
        """
        vec_board, vec_state = state2vec(self.game)

        # Flip perspective when BLACK is to move so that
        # channels 0-1 (white) and 2-3 (black) swap roles.
        if self.game.current_player == BLACK:
            vec_board = vec_board.copy()
            # swap white <-> black channels
            vec_board = np.concatenate([vec_board[2:4], vec_board[0:2]], axis=0)
            # 180° rotation so direction of play is consistent
            vec_board = vec_board[:, ::-1, ::-1].copy()
            vec_state = vec_state.copy()
            vec_state[0] = 1  # always "my turn"
            # Flip chain-taking positions (stored as taken_pos - 1)
            n_chain = len(self.game.chain_taking_moves)
            for i in range(n_chain):
                vec_state[i + 2] = 62 - vec_state[i + 2]

        return vec_board.astype(np.float32), vec_state.astype(np.float32)

    def _compute_reward(self, game_status, player_who_moved, done):
        """Reward from the perspective of the player who just moved."""
        if game_status == GAME_DRAW:
            return self.config.reward_draw
        if game_status == GAME_WHITE_WIN:
            return (
                self.config.reward_win
                if player_who_moved == WHITE
                else self.config.reward_loss
            )
        if game_status == GAME_BLACK_WIN:
            return (
                self.config.reward_win
                if player_who_moved == BLACK
                else self.config.reward_loss
            )
        # Forced draw by step limit
        if done:
            return self.config.reward_draw
        # Game continues
        return self.config.reward_step


class SelfPlayEnv:
    """
    Self-play, ppo gra na samego siebie,
    """

    def __init__(self, config: PPOConfig | None = None):
        self.env = CheckersEnv(config)
        self.config = config or PPOConfig()

    def reset(self):
        obs, mask = self.env.reset()
        return obs, mask

    def step(self, action_id: int):
        return self.env.step(action_id)

    def get_current_player(self):
        return self.env.get_current_player()

    @property
    def n_actions(self):
        return self.env.n_actions
