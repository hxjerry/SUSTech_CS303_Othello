import numba as nb
import numpy as np
import random
from typing import Tuple
import numpy.typing as npt
import numba.types as nbt
import typing
import math
import time
import torch
from .model import OthelloNet
import threading

@nb.njit(nb.uint64(nb.uint64))
def popcount(x : np.uint64) -> np.uint64:
    M1 = np.uint64(0x5555555555555555)
    x = (x & M1) + ((x >> np.uint64(1)) & M1)
    M2 = np.uint64(0x3333333333333333)
    x = (x & M2) + ((x >> np.uint64(2)) & M2)
    M4 = np.uint64(0x0F0F0F0F0F0F0F0F)
    x = (x & M4) + ((x >> np.uint64(4)) & M4)
    M8 = np.uint64(0x00FF00FF00FF00FF)
    x = (x & M8) + ((x >> np.uint64(8)) & M8)
    M16 = np.uint64(0x0000FFFF0000FFFF)
    x = (x & M16) + ((x >> np.uint64(16)) & M16)
    M32 = np.uint64(0x00000000FFFFFFFF)
    x = (x & M32) + ((x >> np.uint64(32)) & M32)
    return x

@nb.njit(nb.uint64(nb.uint64), inline='always')
def lowbit(x : np.uint64) -> np.uint64:
    return x & np.uint64(-x)

@nb.njit(nb.uint64(nb.uint64, nb.uint64))
def candidates(player : np.uint64, opponent : np.uint64) -> np.uint64: # find valid reversi moves on bitboard
    MASK_LEFT = ~np.uint64(0x8080808080808080)
    MASK_RIGHT = ~np.uint64(0x0101010101010101)

    candidates_0 = opponent & ((player & MASK_LEFT) << 1)
    candidates_1 = opponent & ((player & MASK_RIGHT) >> 1)
    candidates_2 = opponent & (player << 8)
    candidates_3 = opponent & (player >> 8)
    candidates_4 = opponent & ((player & MASK_LEFT) << 1 << 8)
    candidates_5 = opponent & ((player & MASK_RIGHT) >> 1 << 8)
    candidates_6 = opponent & ((player & MASK_LEFT) << 1 >> 8)
    candidates_7 = opponent & ((player & MASK_RIGHT) >> 1 >> 8)

    empty = ~(player | opponent)
    moves = np.uint64(0)
    
    for _ in range(6):
        # Direction 0: left
        candidates_0 = (candidates_0 & MASK_LEFT) << 1
        moves |= candidates_0 & empty
        candidates_0 &= opponent
        
        # Direction 1: right
        candidates_1 = (candidates_1 & MASK_RIGHT) >> 1
        moves |= candidates_1 & empty
        candidates_1 &= opponent
        
        # Direction 2: up
        candidates_2 <<= 8
        moves |= candidates_2 & empty
        candidates_2 &= opponent
        
        # Direction 3: down
        candidates_3 >>= 8
        moves |= candidates_3 & empty
        candidates_3 &= opponent
        
        # Direction 4: up-left
        candidates_4 = (candidates_4 & MASK_LEFT) << 1 << 8
        moves |= candidates_4 & empty
        candidates_4 &= opponent
        
        # Direction 5: up-right
        candidates_5 = (candidates_5 & MASK_RIGHT) >> 1 << 8
        moves |= candidates_5 & empty
        candidates_5 &= opponent
        
        # Direction 6: down-left
        candidates_6 = (candidates_6 & MASK_LEFT) << 1 >> 8
        moves |= candidates_6 & empty
        candidates_6 &= opponent
        
        # Direction 7: down-right
        candidates_7 = (candidates_7 & MASK_RIGHT) >> 1 >> 8
        moves |= candidates_7 & empty
        candidates_7 &= opponent
    
    return moves

@nb.njit(nb.types.Tuple((nb.uint64, nb.uint64))(nb.uint64, nb.uint64, nb.uint64))
def move(player : np.uint64, opponent : np.uint64, action : np.uint64) -> Tuple[np.uint64, np.uint64]:
    MASK_LEFT = ~np.uint64(0x8080808080808080)
    MASK_RIGHT = ~np.uint64(0x0101010101010101)

    move_0 = move_1 = move_2 = move_3 = move_4 = move_5 = move_6 = move_7 = action
    line_0 = line_1 = line_2 = line_3 = line_4 = line_5 = line_6 = line_7 = np.uint64(0)
    placed = np.uint64(0)

    for _ in range(7):
        line_0 |= move_0
        move_0 = (move_0 & MASK_LEFT) << 1
        if move_0 & player:
            placed |= line_0
            move_0 = np.uint64(0)
        elif not (move_0 & opponent):
            move_0 = np.uint64(0)
        
        line_1 |= move_1
        move_1 = (move_1 & MASK_RIGHT) >> 1
        if move_1 & player:
            placed |= line_1
            move_1 = np.uint64(0)
        elif not (move_1 & opponent):
            move_1 = np.uint64(0)

        line_2 |= move_2
        move_2 <<= 8
        if move_2 & player:
            placed |= line_2
            move_2 = np.uint64(0)
        elif not (move_2 & opponent):
            move_2 = np.uint64(0)

        line_3 |= move_3
        move_3 >>= 8
        if move_3 & player:
            placed |= line_3
            move_3 = np.uint64(0)
        elif not (move_3 & opponent):
            move_3 = np.uint64(0)

        line_4 |= move_4
        move_4 = (move_4 & MASK_LEFT) << 1 << 8
        if move_4 & player:
            placed |= line_4
            move_4 = np.uint64(0)
        elif not (move_4 & opponent):
            move_4 = np.uint64(0)

        line_5 |= move_5
        move_5 = (move_5 & MASK_RIGHT) >> 1 << 8
        if move_5 & player:
            placed |= line_5
            move_5 = np.uint64(0)
        elif not (move_5 & opponent):
            move_5 = np.uint64(0)

        line_6 |= move_6
        move_6 = (move_6 & MASK_LEFT) << 1 >> 8
        if move_6 & player:
            placed |= line_6
            move_6 = np.uint64(0)
        elif not (move_6 & opponent):
            move_6 = np.uint64(0)   
        
        line_7 |= move_7
        move_7 = (move_7 & MASK_RIGHT) >> 1 >> 8
        if move_7 & player:
            placed |= line_7
            move_7 = np.uint64(0)
        elif not (move_7 & opponent):
            move_7 = np.uint64(0)
    
    player = player | placed
    opponent = opponent & ~placed
    return player, opponent

@nb.njit(nb.uint64(nb.uint64))
def ctz(x : np.uint64) -> np.uint64:
    return popcount((x & (-x)) - np.uint64(1))

def temperature_sample(probs, temperature=1.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    probs = np.asarray(probs, dtype=np.float64)
    if temperature <= 0:
        return int(np.argmax(probs))
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    adjusted = np.exp(logits / temperature)
    adjusted /= adjusted.sum()
    return rng.choice(len(probs), p=adjusted)

class Node:
    LOGIT_MASK = np.array([1 << i for i in range(64)], dtype=np.uint64)

    def __init__(self, player: np.uint64, opponent: np.uint64, model : OthelloNet, policy = None, value = None):
        self.player = player
        self.opponent = opponent
        self.children = []
        self.visits = 1

        moves = np.uint64(candidates(self.player, self.opponent))
        if moves == 0:
            if not candidates(self.opponent, self.player): # terminal
                player_count = popcount(self.player)
                opponent_count = popcount(self.opponent)
                if player_count > opponent_count:
                    self.value = 0.0
                elif player_count < opponent_count:
                    self.value = 1.0
                else:
                    self.value = 0.5
                return 

        if model is not None:
            state_tensor = model.to_tensor(player, opponent)
            state_tensor = state_tensor.to(next(model.parameters()).device)
            policy_logits, value = model(state_tensor)
            self.value = value.item()
            policy_logits = policy_logits[0].to(device = "cpu", dtype=torch.float32).cpu().numpy()
        else:
            self.value = value
            policy_logits = policy
        
        if moves == 0: # pass move
            self.children.append((opponent, player, 1.0))
            return

        policy_logits = policy_logits
        policy_logits -= np.max(policy_logits)
        policy_exp = np.exp(policy_logits)
        policy = policy_exp / np.sum(policy_exp  * ((moves & Node.LOGIT_MASK) > 0).astype(np.float32))

        while moves:
            move_bit = lowbit(moves)
            moves ^= move_bit
            move_bit_index = ctz(move_bit)
            child_player, child_opponent = move(self.player, self.opponent, move_bit)
            child_player, child_opponent = np.uint64(child_player), np.uint64(child_opponent)
            self.children.append((child_opponent, child_player, policy[move_bit_index]))

    def select(self, model: OthelloNet, hashtable: dict) -> float:
        if not self.children:
            self.visits += 1
            return self.value # terminal node

        best_puct = -math.inf
        best_child = None
        best_child_node = None

        self_visits_sqrt = math.sqrt(float(self.visits))

        for (child_opponent, child_player, policy) in self.children:
            reward = 0.
            child_visits = 0.
            child_node = hashtable.get((child_opponent, child_player), None)
            if child_node is not None:
                reward = (child_node.visits - child_node.value) / child_node.visits
                child_visits = float(child_node.visits)
            
            child_puct = reward + 1.0 * policy * self_visits_sqrt / (1.0 + child_visits)
            if child_puct > best_puct:
                best_puct = child_puct
                best_child = (child_opponent, child_player)
                best_child_node = child_node 

        self.visits += 1
        if best_child_node is not None:
            child_score = best_child_node.select(model, hashtable)
        else:
            new_node = Node(best_child[0], best_child[1], model)
            hashtable[best_child] = new_node
            child_score = new_node.value
        self.value += -child_score
        return -child_score

    def select_batch(self, model: OthelloNet, batch_size: int, hashtable: dict):
        selected_leaves = {}
        for _ in range(batch_size):
            node = self
            path = []
            while True:
                path.append(node)
                best_puct = -math.inf
                best_child = None
                best_child_node = None

                node_visits_sqrt = math.sqrt(float(node.visits))

                for (child_opponent, child_player, policy) in node.children:
                    reward = 0.
                    child_visits = 0.
                    child_node = hashtable.get((child_opponent, child_player), None)
                    if child_node is not None:
                        reward = (child_node.visits - child_node.value) / child_node.visits
                        child_visits = float(child_node.visits)
                    if (child_opponent, child_player) in selected_leaves:
                        reward = -1e9 # already selected in this batch

                    child_puct = reward + 1.0 * policy * node_visits_sqrt / (1.0 + child_visits)
                    if child_puct > best_puct:
                        best_puct = child_puct
                        best_child = (child_opponent, child_player)
                        best_child_node = child_node
                
                if best_child_node is None or not best_child_node.children: # Expandable or terminal
                    break
                node = best_child_node
            
            if best_child in selected_leaves:
                continue
            selected_leaves[best_child] = path
            for node in path:
                node.value += 1.0 # Apply virtual loss, positive to disencourage parent nodes

        # Processes already expanded leaves
        expand_leaves = []
        for (child_opponent, child_player), path in selected_leaves.items():
            if (child_opponent, child_player) in hashtable:
                leaf_node = hashtable[(child_opponent, child_player)]
            else:
                expand_leaves.append((child_opponent, child_player, path))
                continue
            score = leaf_node.value
            for node in reversed(path):
                score = -score
                node.visits += 1
                node.value += score - 1.0 # Remove virtual loss and add real score

        if not expand_leaves:
            return 0

        # Collect tensors for batch inference
        boards = []
        for (child_opponent, child_player, path) in expand_leaves:
            boards.append(model.to_tensor(child_opponent, child_player))
        boards_tensor = torch.cat(boards, dim=0)
        boards_tensor = boards_tensor.to(next(model.parameters()).device)
        policy_logits, values = model(boards_tensor)
        policy_logits = policy_logits.to(device = "cpu", dtype=torch.float32).cpu().numpy()
        values = values.to(device = "cpu", dtype=torch.float32).cpu().numpy()

        for i, (child_opponent, child_player, path) in enumerate(expand_leaves):
            hashtable[(child_opponent, child_player)] = Node(child_opponent, child_player, None, policy_logits[i], values[i].item())
            score = values[i].item()
            for node in reversed(path):
                score = -score
                node.visits += 1
                node.value += score - 1.0 # Remove virtual loss and add real score
        return len(expand_leaves)

    # return the needed data to incoporate this node into training data
    # (player, opponent, children exploration probabilities)
    def snapshot(self, hashtable: dict):
        policy = np.zeros(64, dtype=np.float32)
        policy[:] = -1
        for (child_opponent, child_player, _) in self.children:
            move = (self.player | self.opponent) ^ (child_player | child_opponent)
            if not move:
                continue
            move_index = ctz(move)
            if (child_opponent, child_player) in hashtable:
                policy[move_index] = hashtable[(child_opponent, child_player)].visits
            else:
                policy[move_index] = 0.0

        total_visits = np.sum(policy[policy >= 0])
        if total_visits > 0:
            policy[policy >= 0] /= total_visits
        
        return (self.player, self.opponent, policy)

    # Inject noise to the current node
    # Use on root when generating training data
    def inject_noise(self, epsilon: float, C: float):
        if len(self.children) <= 1:
            return
        num_children = len(self.children)
        alpha = C / num_children
        noise = np.random.dirichlet([alpha] * num_children)
        for i in range(num_children):
            (child_opponent, child_player, policy) = self.children[i]
            new_policy = (1 - epsilon) * policy + epsilon * noise[i]
            self.children[i] = (child_opponent, child_player, new_policy)

    # select optimal move based on visit counts
    def select_optimal(self, model: OthelloNet, hashtable: dict, temperature: float = 0.0):
        best_visits = -1
        best_child = None

        counts = []
        for (child_opponent, child_player, _) in self.children:
            child_node = hashtable.get((child_opponent, child_player), None)
            if child_node is not None:
                counts.append(child_node.visits)
            else:
                counts.append(0)
        counts = np.array(counts, dtype=np.float64)
        selected_index = temperature_sample(counts / counts.sum(), temperature)
        best_child = self.children[selected_index][:2]

        if best_child not in hashtable:
            new_node = Node(best_child[0], best_child[1], model)
            hashtable[best_child] = new_node

        return hashtable[best_child]

    def result(self) -> float:
        player_count = popcount(self.player)
        opponent_count = popcount(self.opponent)
        if player_count < opponent_count:
            return 1.0
        elif player_count > opponent_count:
            return 0.0
        else:
            return 0.5

def reconstruct_hashtable(new_root: Node, hashtable: dict):
    new_hashtable = {}
    def dfs(node: Node):
        if (node.player, node.opponent) in new_hashtable:
            return
        new_hashtable[(node.player, node.opponent)] = node
        for (child_opponent, child_player, _) in node.children:
            if (child_opponent, child_player) in hashtable:
                dfs(hashtable[(child_opponent, child_player)])
    dfs(new_root)
    return new_hashtable

def starting_node(model: OthelloNet):
    init_player = np.uint64(0x0000000810000000)
    init_opponent = np.uint64(0x0000001008000000)
    root = Node(init_player, init_opponent, model)
    hashtable = {}
    hashtable[(init_player, init_opponent)] = root
    return root, hashtable

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OthelloNet()
    model.to(device)
    model.eval()

    import time
    # import cProfile

    with torch.inference_mode():
        compiled_model = torch.compile(model, mode="reduce-overhead", dynamic=True)
        
        root, hashtable = starting_node(compiled_model)

        t0 = time.time()
        def profile_mcts():
            for _ in range(1024):
                root.select_batch(compiled_model, 32, hashtable)
        # cProfile.run('profile_mcts()', sort='cumtime')
        profile_mcts()
        t1 = time.time()
        print(f"MCTS time: {t1 - t0} seconds")
        print(f"{root.visits} {root.value}")
        # root.select(compiled_model, hashtable)

