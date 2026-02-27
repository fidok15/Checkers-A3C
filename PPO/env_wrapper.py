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
    state2vec, action2id, N_ACTION_64,
)
from deepdraughts.env.py_env.piece import Piece

from config import PPOConfig


class CheckersEnv:
    """
    Bierzemy obserwacje z `state2vec`:
      - vec_board: np.array of shape (4, 8, 8)
      - vec_state: np.array of shape (19,)

    Akcje to integery w [0, N_ACTION_64)  (280 possible moves).
    W każdym ruchu dostępna jest tylko część możliwych ruchów;  `get_action_mask()`.

    Agent zawsze widzi plansze jakby grał białymi, jeśli gra czarnymi,
    to wrapper odwraca planszę i cechy tak, żeby się zgadzały jakby grał białymi.
    """

    def __init__(self, config: PPOConfig | None = None):
        self.config = config or PPOConfig()
        self.game: Game | None = None
        self.n_actions = N_ACTION_64  # 280
        self.step_count = 0

        # Mapping from action_id -> Move for current legal moves
        self._legal_map: dict[int, object] = {}


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
        """Rebuild the mapping action_id -> Move for current legal moves."""
        self._legal_map = {}
        moves = self.game.get_all_available_moves()
        for m in moves:
            aid = action2id(m)
            self._legal_map[aid] = m

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
            # flip board vertically so direction of play is consistent
            vec_board = vec_board[:, ::-1, :].copy()
            vec_state = vec_state.copy()
            vec_state[0] = 1  # always "my turn"

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
