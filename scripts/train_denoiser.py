from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import torch
from fusion.learned.denoiser import Denoiser
from fusion.learned.datasets import load_denoiser_dataset
from fusion.learned.train import train
from fusion.learned.checkpoints import save_checkpoint
import matplotlib.pyplot as plt

# Load train/validation data and norm stats from prior created dataset
train_dataset, val_dataset, normalization_stats = load_denoiser_dataset(path='data/processed/denoiser_dataset.npz', val_fraction=0.20, seed=42)

# Wrap dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False) # Shuffling unnecessary for val set

model = Denoiser(dim_z=config['model']['dim_z'], 
                 dim_pos=config['model']['dim_pos'], 
                 hidden_channels=config['model']['hidden_channels'], 
                 n_layers=config['model']['n_layers'], 
                 kernel_size=config['model']['kernel_size'])

# Save history after training/evaluating the model
history = train(model, train_loader, val_loader, config)

# Save this to processed data
checkpoint_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "denoiser_checkpoint.pt"
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
plt.savefig("figures/train_denoiser.png")
plt.show()