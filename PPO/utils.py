import os
import time
import numpy as np
import torch
from collections import deque


class GameStats:
    """Track win/loss/draw statistics over a sliding window."""

    def __init__(self, window: int = 100):
        self.window = window
        self.results = deque(maxlen=window)  # 1=win_white, -1=win_black, 0=draw
        self.game_lengths = deque(maxlen=window)

    def record(self, winner: int, length: int):
        self.results.append(winner)
        self.game_lengths.append(length)

    @property
    def n_games(self):
        return len(self.results)

    @property
    def white_win_rate(self):
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r == 1) / len(self.results)

    @property
    def black_win_rate(self):
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r == -1) / len(self.results)

    @property
    def draw_rate(self):
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r == 0) / len(self.results)

    @property
    def avg_length(self):
        if not self.game_lengths:
            return 0.0
        return np.mean(self.game_lengths)

    def summary(self) -> str:
        return (
            f"Games: {self.n_games} | "
            f"White: {self.white_win_rate:.1%} | "
            f"Black: {self.black_win_rate:.1%} | "
            f"Draw: {self.draw_rate:.1%} | "
            f"Avg len: {self.avg_length:.0f}"
        )


class Timer:
    """Simple wall-clock timer."""

    def __init__(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time

    def elapsed_str(self) -> str:
        s = self.elapsed()
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = int(s % 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
