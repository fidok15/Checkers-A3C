import sys
import os

# --- HACK NA ŚCIEŻKI ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir)) 
sys.path.append(root_dir)
sys.path.append(os.path.dirname(current_dir))

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import time
import csv
import numpy as np

# Obsługa importu w zależności od struktury folderów
try:
    from deepdraughts.env.py_env.env_utils import get_env_args
except ImportError:
    from deepdraughts.env import get_env_args

from deepdraughts.env import Game, game_is_over, game_winner
from deepdraughts.net_pytorch import PolicyValueNet
from shared_adam import SharedAdam

env = Game()
vec_board, vec_state = env.to_vector()
print(type(vec_board))
print(vec_board.shape if hasattr(vec_board, 'shape') else len(vec_board))