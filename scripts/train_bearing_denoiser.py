from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import torch
import torch.nn.functional as F
from fusion.learned.denoiser import Denoiser
from fusion.learned.datasets import load_denoiser_dataset
from fusion.learned.train import train
from fusion.learned.checkpoints import save_checkpoint
import matplotlib.pyplot as plt

# Set seed explicitly
torch.manual_seed(config['seed'])

# Load train/validation data and norm stats from prior created dataset
train_dataset, val_dataset, normalization_stats = load_denoiser_dataset(path='data/processed/bearing_denoiser_dataset.npz', val_fraction=config['dataset']['val_fraction'], seed=config['seed'])

# Wrap dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False) # Shuffling unnecessary for val set

model = Denoiser(dim_z=config['model_bearing']['dim_z'], 
                 dim_pos=config['model_bearing']['dim_pos'], 
                 hidden_channels=config['model_bearing']['hidden_channels'], 
                 n_layers=config['model_bearing']['n_layers'], 
                 kernel_size=config['model_bearing']['kernel_size'])

# MSE loss function
def mse_loss_fn(output, targets):
    return F.mse_loss(output, targets)

# Save history after training/evaluating the model
history = train(model, train_loader, val_loader, config, mse_loss_fn)

# Save this to processed data (Create copy of config with model_bearing instead of model)
save_config = config.copy()
save_config['model'] = config['model_bearing']

checkpoint_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "bearing_denoiser_checkpoint.pt"
save_checkpoint(path=checkpoint_path, 
                model=model, 
                normalization_stats=normalization_stats, 
                config=save_config)

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
plt.savefig("figures/train_bearing_denoiser.png")
plt.show()