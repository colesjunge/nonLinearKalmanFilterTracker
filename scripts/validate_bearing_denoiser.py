from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
import torch
from fusion.learned.checkpoints import load_checkpoint
from fusion.learned.denoiser import build_denoiser
from fusion.learned.datasets import load_denoiser_dataset

# Compute sigma_pos empirically for bearing denoiser

model, normalization_stats, checkpoint_config = load_checkpoint(path='data/processed/bearing_denoiser_checkpoint.pt', 
                                                                build_model_fn=build_denoiser)

model.eval()

seed = checkpoint_config["seed"]
val_fraction = checkpoint_config['dataset']['val_fraction']

_, val_dataset, _ = load_denoiser_dataset(
         path='data/processed/bearing_denoiser_dataset.npz',
         val_fraction=val_fraction, seed=seed)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False) # Shuffling unnecessary for val set

errors = []
with torch.no_grad():
    for windows, targets in val_loader:
        predictions = model(windows) # Forward pass
        errors.append(predictions-targets) 

# Error is (batch_size, dim_pos) to become (n_val_samples, dim_pos)
all_errors = torch.cat(errors, dim=0)

# Calculate mean_bias, std, and sigma-Pos
mean_bias = all_errors.mean(dim=0) # (dim_pos,)
std = all_errors.std(dim=0) # (dim_pos,)
sigma_pos = torch.sqrt(torch.mean(std ** 2)).item()

print(f"Bias Mean: {mean_bias}") # [ 0.9880, -2.6997]
print(f"Bias Standard deviation: {std}") # [38.3096, 38.1223]
print(f"Sigma_pos: {sigma_pos}") # 38.21607971191406

