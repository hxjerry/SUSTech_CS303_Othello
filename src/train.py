from .playout import ConcurrentPlayoutPool
from .model import OthelloNet
import torch
from collections import deque
import random
from tqdm import tqdm
import numpy as np
import copy
import sys
import os
import datetime

def enhance_data(boards, policies, eval):
    batch_size = boards.shape[0]
    enhanced_boards = np.zeros((batch_size * 8, 2, 8, 8), dtype=boards.dtype)
    enhanced_policies = np.zeros((batch_size * 8, 64), dtype=policies.dtype)
    enhanced_eval = np.zeros((batch_size * 8,), dtype=eval.dtype)

    boards = boards.copy()
    policies = policies.copy().reshape(-1, 8, 8)
    eval = eval.copy()

    for i in range(2):
        for j in range(4):
            index = i * 4 + j
            boards = np.rot90(boards, k=1, axes=(2, 3))
            policies = np.rot90(policies, k=1, axes=(1, 2))

            enhanced_boards[index:index + batch_size] = boards
            enhanced_policies[index:index + batch_size] = policies.reshape(-1, 64)
            enhanced_eval[index:index + batch_size] = eval

        boards = np.flip(boards, axis=3)
        policies = np.flip(policies, axis=2)

    return enhanced_boards, enhanced_policies, enhanced_eval

if __name__ == "__main__":
    date_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    ckpt_dir = f"checkpoints/{date_str}"
    os.makedirs(ckpt_dir, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = OthelloNet()
    model.to(device)

    if len(sys.argv) > 1:
        restart_ckpt = sys.argv[1].strip()
        print(f"Restarting from checkpoint: {restart_ckpt}")
        model.load_state_dict(torch.load(restart_ckpt, map_location=device))

    # Separate parameters for weight decay
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue # Skip frozen parameters

        if "bn" in name or "bias" in name: # Exclude BN parameters and biases
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": 1e-4},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=1e-3,
    )

    phases = 128
    threads = 12

    minibatches_per_phase = 16
    playouts_per_minibatch = 2

    batch_size = 512

    buffer_phases = 4
    
    replay_buffer = deque(maxlen=minibatches_per_phase * playouts_per_minibatch * threads * buffer_phases)

    pool = ConcurrentPlayoutPool(num_threads=threads)

    for phase in tqdm(range(phases), desc="Phases"):

        old_model_dict = copy.deepcopy(model.state_dict())
        old_optimizer_dict = copy.deepcopy(optimizer.state_dict())

        for _ in (batch_pbar := tqdm(range(minibatches_per_phase), desc=f"Batches")):
            replay_buffer.extend(pool.playout(model, rounds_per_thread=playouts_per_minibatch))

            batch = random.sample(replay_buffer, min(len(replay_buffer), batch_size))
            boards = np.concatenate([data[0] for data in batch], axis=0)
            policies = np.concatenate([data[1] for data in batch], axis=0)
            eval = np.concatenate([data[2] for data in batch], axis=0)
            
            boards = torch.tensor(boards, dtype=torch.float32, device=device)
            policies = torch.tensor(policies, dtype=torch.float32, device=device)
            eval = torch.tensor(eval, dtype=torch.float32, device=device)

            pred_policies, pred_eval = model(boards)

            invalid_mask = policies < 0
            pred_policies = pred_policies.masked_fill(invalid_mask, float('-1e9'))
            pred_policies = torch.log_softmax(pred_policies, dim=1)
            policies = policies.masked_fill(invalid_mask, 0.0)
            loss_policy = - (policies * pred_policies).sum(dim=1).mean()

            loss_value = torch.nn.functional.mse_loss(pred_eval.view(-1), eval)
            loss = loss_policy + loss_value
            (loss  * len(batch) / batch_size).backward()
            batch_pbar.set_postfix(Loss = f"{loss.item():.4f}")

            with open(f"{ckpt_dir}/training_log.txt", "a") as f:
                f.write(f"Phase {phase + 1}, Loss: {loss.item():.4f}, Policy Loss: {loss_policy.item():.4f}, Value Loss: {loss_value.item():.4f}\n")

            optimizer.step()
            optimizer.zero_grad()

        new_model_dict = model.state_dict()
        score = pool.benchmark(new_model_dict, old_model_dict, rounds_per_thread=4)
        tqdm.write(f"Phase {phase + 1}, Epoch completed. New model vs old model score: {score:.4f}")

        if score < 0.10: # Revert to old model
            model.load_state_dict(old_model_dict)
            optimizer.load_state_dict(old_optimizer_dict)
            tqdm.write(f"Phase {phase + 1}, Model reverted.")
            torch.save(model.state_dict(), f"{ckpt_dir}/model_phase_{phase + 1}_reverted.pth")
        else:
            tqdm.write(f"Phase {phase + 1}, Model accepted.")
            torch.save(model.state_dict(), f"{ckpt_dir}/model_phase_{phase + 1}.pth")

    pool.close()
