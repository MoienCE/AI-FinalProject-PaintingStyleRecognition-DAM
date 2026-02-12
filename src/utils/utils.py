import random
import os
import numpy as np
import torch
import json

def set_seed(seed=42):
    """Sets the seed for reproducibility across runs."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    print(f"🌱 Seed set to {seed}")

def save_metrics(metrics, path):
    """Saves metrics dictionary to a JSON file."""
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=4)

def load_config(config_path):
    """Loads configuration from a JSON file (optional for later)."""
    with open(config_path, 'r') as f:
        return json.load(f)