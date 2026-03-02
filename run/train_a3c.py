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

# --- PARAMETRY TRENINGU (DOSTROIŁEM POD TYDZIEŃ NAUKI) ---
SAVE_DIR = "./savedata_a3c"
CHECKPOINT_FILE = "a3c_draughts_checkpoint.pth"
LOG_FILE = "training_log.csv"

# Wyświetlanie w konsoli co 50 gier (żeby nie spamowało)
PRINT_INTERVAL = 50 

NUM_WORKERS = mp.cpu_count()
MAX_EPISODES = 50000000      
SAVE_INTERVAL_SEC = 600      
UPDATE_GLOBAL_ITER = 20      
GAMMA = 0.99                  
ENTROPY_BETA = 0.05          
LR = 0.0001                  

class Worker(mp.Process):
    def __init__(self, global_net, optimizer, update_lock, global_ep, global_ep_r, res_queue, name, env_args):
        super(Worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.g_net = global_net
        self.opt = optimizer
        self.update_lock = update_lock  # POPRAWKA: przechowujemy lock
        
        # Inicjalizacja gry - jeśli env_args to krotka (nsize, ...), przekaż do Game
        # Zakładam, że Game przyjmuje odpowiednie argumenty; jeśli nie, zostaw puste
        self.env = Game()  # ewentualnie Game(*env_args) jeśli wymaga
        
        nsize, _, n_states, n_actions = env_args
        self.env_net_params = (nsize, n_states, n_actions)
        self.l_net = PolicyValueNet(nsize, n_states, n_actions)

    def run(self):
        torch.set_num_threads(1)
        total_step = 1
        
        while self.g_ep.value < MAX_EPISODES:
            # ładujemy globalne wagi na początku epizodu
            self.l_net.load_state_dict(self.g_net.state_dict())
            self.env = Game()  # Reset gry (lub Game(*env_args))
            
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            
            while True:
                # 1. Pobranie stanu
                vec_board, vec_state = self.env.to_vector()
                s_board = torch.tensor(vec_board, dtype=torch.float).unsqueeze(0)
                s_state = torch.tensor(vec_state, dtype=torch.float).unsqueeze(0)

                # 2. Wybór akcji
                logits, _ = self.l_net(s_board, s_state)
                legal_moves = self.env.get_all_available_moves()
                legal_ids = [m.id() for m in legal_moves]
                
                # Maskowanie nielegalnych ruchów
                mask = torch.full_like(logits, -float('inf'))
                mask[0, legal_ids] = 0
                masked_logits = logits + mask
                
                probs = F.softmax(masked_logits, dim=1)
                action_idx = torch.multinomial(probs, 1).item()
                move_obj = next(m for m in legal_moves if m.id() == action_idx)

                # 3. Wykonanie ruchu
                game_status = self.env.do_move(move_obj)
                
                r = 0
                done = game_is_over(game_status)
                
                if done:
                    winner = game_winner(game_status)
                    if winner == 1 or winner == 2:  # ktoś wygrał

                        if winner != self.env.current_player:
                            r = 1.0
                        else:
                            r = -1.0
                    else:
                        r = 0.0

                ep_r += r
                buffer_s.append((s_board, s_state))
                buffer_a.append(action_idx)
                buffer_r.append(r)

                # 4. Aktualizacja (Update)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    if done:
                        v_s_next = 0
                    else:
                        vec_board_next, vec_state_next = self.env.to_vector()
                        sb_next = torch.tensor(vec_board_next, dtype=torch.float).unsqueeze(0)
                        ss_next = torch.tensor(vec_state_next, dtype=torch.float).unsqueeze(0)
                        _, v_s_next = self.l_net(sb_next, ss_next)
                        v_s_next = v_s_next.item()

                    v_target = v_s_next
                    buffer_v_target = []
                    
                    # Obliczanie celu wartości dla gier turowych
                    for r_step in buffer_r[::-1]:
                        v_target = r_step - GAMMA * v_target
                        buffer_v_target.append(v_target)
                    buffer_v_target.reverse()

                    # Batchowanie
                    bs_board = torch.cat([x[0] for x in buffer_s])
                    bs_state = torch.cat([x[1] for x in buffer_s])
                    ba = torch.tensor(buffer_a).view(-1, 1)
                    bvt = torch.tensor(buffer_v_target, dtype=torch.float).view(-1, 1)

                    # Forward pass lokalnej sieci
                    logits, values = self.l_net(bs_board, bs_state)
                    
                    log_probs = F.log_softmax(logits, dim=1)
                    log_prob_a = log_probs.gather(1, ba)
                    advantage = bvt - values.detach()
                    
                    policy_loss = -(log_prob_a * advantage).mean()
                    value_loss = F.mse_loss(values, bvt)
                    
                    probs_all = F.softmax(logits, dim=1)
                    entropy = -(probs_all * log_probs).sum(1).mean()
                    
                    total_loss = policy_loss + value_loss - ENTROPY_BETA * entropy

                    # Backward na lokalnej sieci (nie wymaga blokady)
                    total_loss.backward()
                    
                    # Gradient clipping na lokalnej sieci
                    torch.nn.utils.clip_grad_norm_(self.l_net.parameters(), 0.5)
                    
                    with self.update_lock:
                        # Zerujemy gradienty globalne
                        self.opt.zero_grad()
                        # Kopiujemy gradienty z lokalnej sieci do globalnej
                        for lp, gp in zip(self.l_net.parameters(), self.g_net.parameters()):
                            gp._grad = lp.grad
                        # Wykonujemy krok optymalizatora na globalnych wagach
                        self.opt.step()
            
                    # Czyścimy bufory
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:
                        with self.g_ep.get_lock():
                            self.g_ep.value += 1
                            curr_ep = self.g_ep.value
                        
                        with self.g_ep_r.get_lock():
                            if self.g_ep_r.value == 0:
                                self.g_ep_r.value = ep_r
                            else:
                                self.g_ep_r.value = self.g_ep_r.value * 0.99 + ep_r * 0.01
                        
                        self.res_queue.put((curr_ep, ep_r, total_loss.item(), entropy.item()))
                        
                        if curr_ep % PRINT_INTERVAL == 0:
                            print(f"{self.name} | Ep: {curr_ep} | Reward: {ep_r:.0f} | AvgR: {self.g_ep_r.value:.2f} | Loss: {total_loss.item():.2f} | Ent(Explor): {entropy.item():.2f}")
                        break
                
                total_step += 1

def save_checkpoint(model, optimizer, episode, filepath):
    print(f"Saving checkpoint to {filepath}...")
    torch.save({
        'episode': episode,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, filepath)

def load_checkpoint(filepath, model, optimizer):
    if os.path.isfile(filepath):
        print(f"Loading checkpoint from {filepath}...")
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint['episode']
    else:
        print("No checkpoint found, starting from scratch.")
        return 0

if __name__ == "__main__":
    #tworzy nowy czysty proces
    mp.set_start_method('spawn', force=True)

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    checkpoint_path = os.path.join(SAVE_DIR, CHECKPOINT_FILE)
    log_path = os.path.join(SAVE_DIR, LOG_FILE)
    
    if not os.path.exists(log_path):
        with open(log_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Episode", "Reward", "Loss", "Entropy", "Timestamp"])
    #wymiar, _ ,19 plaszczyzn gdzie sa pionki, liczba wszystkich ruchow 
    env_args = get_env_args()
    nsize, _, n_states, n_actions = env_args
    
    global_net = PolicyValueNet(nsize, n_states, n_actions)

    #pamiec wspoldzieloan 
    global_net.share_memory() 
    optimizer = SharedAdam(global_net.parameters(), lr=LR)
    #lock updateow 
    update_lock = mp.Lock()

    #zaldaowanie chackpointa 
    start_episode = load_checkpoint(checkpoint_path, global_net, optimizer)
    
    #zmienne globalne epizody, suma nagrod, kolejka procesow 
    global_ep = mp.Value('i', start_episode)
    global_ep_r = mp.Value('d', 0.)
    res_queue = mp.Queue()

    #stworzenie listy pracownikow 
    workers = [Worker(global_net, optimizer, update_lock, global_ep, global_ep_r, res_queue, i, env_args) 
           for i in range(NUM_WORKERS)]
    
    #uruchomienie ich
    [w.start() for w in workers]

    try:
        last_save_time = time.time()
        while True:
            # Zbieranie logów
            while not res_queue.empty():
                try:
                    data = res_queue.get_nowait()
                    ep, r, loss, ent = data
                    with open(log_path, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([ep, r, loss, ent, time.strftime("%Y-%m-%d %H:%M:%S")])
                except:
                    break

            if time.time() - last_save_time > SAVE_INTERVAL_SEC:
                with update_lock:
                    save_checkpoint(global_net, optimizer, global_ep.value, checkpoint_path)
                last_save_time = time.time()
            
            alive_workers = [w.is_alive() for w in workers]
            if not any(alive_workers):
                print("All workers finished.")
                break
            
            time.sleep(1) 

    except KeyboardInterrupt:
        print("Stopping training manually...")
        with update_lock:
            save_checkpoint(global_net, optimizer, global_ep.value, checkpoint_path)
    
    [w.join() for w in workers]

    