from . import game
from .model import OthelloNet
import numpy as np
import os
import multiprocessing as mp
import torch
import gc

def playout(model: game.OthelloNet):
    player0_data = []
    player1_data = []
    
    player = 0
    steps = 0
    root, hashtable = game.starting_node(model)
    while root.children:
        root.inject_noise(epsilon=0.25, C=6) # Add Dirichlet noise for exploration
        for _ in range(32):
            root.select_batch(model, 32, hashtable)
        data = root.snapshot(hashtable)
        if player == 0:
            player0_data.append(data)
        else:
            player1_data.append(data)
        root = root.select_optimal(model, hashtable, temperature=1.0 if steps < 6 else 0.0)
        hashtable = game.reconstruct_hashtable(root, hashtable)
        player = 1 - player
        steps += 1

    data = []
    result = root.result()
    if player == 1:
        player0_data, player1_data = player1_data, player0_data
    
    eval = np.array([result] * len(player0_data) + [1 - result] * len(player1_data), dtype=np.float32)
    boards = np.array([[d[0], d[1]] for d in player0_data + player1_data], dtype=np.uint64).reshape(-1, 2, 1, 1)
    MASK = np.array([1 << i for i in range(64)], dtype=np.uint64).reshape(1, 1, 8, 8)
    boards_tensor = ((boards & MASK) != 0).astype(np.uint8)
    policies = np.array([d[2] for d in player0_data + player1_data], dtype=np.float32)

    return boards_tensor, policies, eval

def compare_models(model: game.OthelloNet, param_a, param_b, rounds: int):
    def play(param_a, param_b):
        current_board = (np.uint64(0x0000000810000000), np.uint64(0x0000001008000000))
        
        hashtable_a = {}
        hashtable_b = {}

        moves = 0

        while True:
            model.load_state_dict(param_a)
            if current_board not in hashtable_a:
                hashtable_a[current_board] = game.Node(current_board[0], current_board[1], model)
            hashtable_a = game.reconstruct_hashtable(hashtable_a[current_board], hashtable_a)
            for _ in range(32):
                hashtable_a[current_board].select_batch(model, 32, hashtable_a)
            root_b = hashtable_a[current_board].select_optimal(model, hashtable_a, temperature=1.0 if moves < 6 else 0.0)
            current_board = (root_b.player, root_b.opponent)
            if not root_b.children:
                current_board = (current_board[1], current_board[0]) # a is always first when comparing
                break
            moves += 1

            model.load_state_dict(param_b)
            if current_board not in hashtable_b:
                hashtable_b[current_board] = game.Node(current_board[0], current_board[1], model)
            hashtable_b = game.reconstruct_hashtable(hashtable_b[current_board], hashtable_b)
            for _ in range(32):
                hashtable_b[current_board].select_batch(model, 32, hashtable_b)
            root_a = hashtable_b[current_board].select_optimal(model, hashtable_b, temperature=1.0 if moves < 6 else 0.0)
            current_board = (root_a.player, root_a.opponent)
            if not root_a.children:
                break
            moves += 1
        
        a_count = game.popcount(current_board[0])
        b_count = game.popcount(current_board[1])
        if a_count < b_count:
            return 1
        elif a_count > b_count:
            return -1
        else:
            return 0
    
    score = 0
    for _ in range(rounds):
        score += play(param_a, param_b)
        score -= play(param_b, param_a) # Ensure fairness by switching starting player
    return score

model = None
def initialize_worker():
    with torch.inference_mode():
        global model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = OthelloNet()
        model.eval()
        model.to(device)
        with torch.inference_mode(): #, torch.autocast(device_type="cuda"):
            model.compile(mode="reduce-overhead")

def selfplay_worker(params):
    gc.collect()
    with torch.inference_mode():
        global model
        model_dict, rounds = params
        model.load_state_dict(model_dict)
        games_data = []
        for _ in range(rounds):
            games_data.append(playout(model))
        return games_data

def benchmark_worker(params):
    gc.collect()
    with torch.inference_mode():
        global model
        param_a, param_b, rounds = params
        score = compare_models(model, param_a, param_b, rounds)
        return score

mp.set_start_method("spawn", force=True)

class ConcurrentPlayoutPool:
    def __init__(self, num_threads: int):
        self.num_threads = num_threads
        self.pool = mp.Pool(processes=num_threads, initializer=initialize_worker)

    def playout(self, model: game.OthelloNet, rounds_per_thread: int):
        model_dict = model.state_dict()
        results = self.pool.map(selfplay_worker, [(model_dict, rounds_per_thread) for _ in range(self.num_threads)])
        games_data = []
        for res in results:
            games_data.extend(res)
        return games_data
    
    def benchmark(self, param_a, param_b, rounds_per_thread: int):
        model_dict_a = param_a
        model_dict_b = param_b
        results = self.pool.map(benchmark_worker, [(model_dict_a, model_dict_b, rounds_per_thread) for _ in range(self.num_threads)])
        return sum(results) / self.num_threads / 2 / rounds_per_thread # Average score per game
    
    def close(self):
        self.pool.close()
        self.pool.join()

if __name__ == "__main__":
    with torch.inference_mode():
        model = OthelloNet()
        params = model.state_dict()

    pool = ConcurrentPlayoutPool(num_threads=12)
    average_score = pool.benchmark(params, params, rounds_per_thread=4)
    print(f"Average score when playing against itself: {average_score}")
    pool.close()