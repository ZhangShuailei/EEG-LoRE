import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("medium")


def cosine_schedule(base_value, final_value, epochs, steps_per_epoch, warmup_epochs=0):
    total_steps = epochs * steps_per_epoch
    warmup_steps = min(max(warmup_epochs, 0) * steps_per_epoch, total_steps)
    warmup = np.linspace(0.0, base_value, warmup_steps, endpoint=True) if warmup_steps else np.array([])
    remaining = total_steps - warmup_steps
    if remaining <= 0:
        return warmup
    steps = np.arange(remaining)
    decay = final_value + 0.5 * (base_value - final_value) * (1.0 + np.cos(np.pi * steps / remaining))
    return np.concatenate((warmup, decay))


def save_state(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
