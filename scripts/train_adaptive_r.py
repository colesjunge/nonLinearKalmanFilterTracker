from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import torch
from fusion.learned.covariance import gaussian_nll, AdaptiveDenoiser
from fusion.learned.denoiser import build_denoiser
from fusion.learned.datasets import load_denoiser_dataset
from fusion.learned.train import train
from fusion.learned.checkpoints import save_checkpoint, load_checkpoint
import matplotlib.pyplot as plt

# Set seed explicitly
torch.manual_seed(config['seed'])

# Load denoiser model (used in train_denoiser.py) for appropriate comparison later on 
# Also use norm stats and checkpoint_config for all applicable (sigma_scale and min_diag come from global config)
denoiser, normalization_stats, checkpoint_config = load_checkpoint(path="data/processed/denoiser_checkpoint.pt", build_model_fn=build_denoiser)

# Load train/validation data from prior created dataset
train_dataset, val_dataset, _ = load_denoiser_dataset(path='data/processed/denoiser_dataset.npz', val_fraction=checkpoint_config['dataset']['val_fraction'], seed=checkpoint_config['seed'])

# Wrap dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False) # Shuffling unnecessary for val set

# Define adaptive denoiser
model = AdaptiveDenoiser(denoiser=denoiser,
                         sigma_scale=config['covariance']['sigma_scale'],
                         min_diag=config['covariance']['min_diag'])

# Gaussian NLL loss function
def nll_loss_fn(output, targets):
    pos, L = output
    return gaussian_nll(pos, targets, L)

# Save history after training/evaluating the model
history = train(model, train_loader, val_loader, config, nll_loss_fn)

# Save this to processed data
checkpoint_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "adaptive_r_checkpoint.pt"
save_checkpoint(path=checkpoint_path, 
                model=model, 
                normalization_stats=normalization_stats, 
                config=config)

print(f"Saved checkpoint to {checkpoint_path}")

# Plot Train Loss / Val Loss vs Epochs
epoch_range = range(len(history['train_loss']))

plt.figure(figsize=(10, 6))
plt.plot(
    epoch_range,
    history['train_loss'],
    label="Train Loss"
)
plt.plot(
    epoch_range,
    history['val_loss'],
    label="Validation Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train/Val Loss across Epochs")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/train_adaptive_r.png")
plt.show()