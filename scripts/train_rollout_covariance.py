from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import torch
import numpy as np
from fusion.learned.covariance import gaussian_nll
from fusion.learned.denoiser import build_denoiser
from fusion.learned.datasets import RolloutWindowDataset, load_rollout_window_dataset, compute_warm_states
from fusion.learned.train import train
from fusion.learned.checkpoints import save_checkpoint, load_checkpoint
from tracking.models.measurement import LinearPosition2D
from tracking.models.motion import ConstantVelocity2D
from fusion.learned.covariance import RolloutCovarianceModel
import matplotlib.pyplot as plt


# Set seed explicitly
torch.manual_seed(config['seed'])

# Load denoiser model (used in train_denoiser.py) for appropriate comparison later on 
# Also use norm stats and checkpoint_config for all applicable (sigma_scale and min_diag come from global config)
denoiser, normalization_stats, checkpoint_config = load_checkpoint(path="data/processed/denoiser_checkpoint.pt", build_model_fn=build_denoiser)

K = config['rollout']['K']
mean = normalization_stats['mean']
std = normalization_stats['std']
val_fraction = checkpoint_config['dataset']['val_fraction']
seed = checkpoint_config['seed']

# Load train/validation data from prior created dataset
train_dataset, val_dataset, _ = load_rollout_window_dataset(path='data/processed/denoiser_dataset.npz', val_fraction=val_fraction, seed=seed, K=K, mean=mean, std=std)

# Wrap dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False) # Shuffling unnecessary for val set

# Build F, H, Q matrices for differentiable KF
motion_model = ConstantVelocity2D(sigma_a=0.0)
measurement_model = LinearPosition2D(sigma_pos=1.0) # Sigma_pos is arbitrary here

dummy_state = np.zeros(4, dtype=np.float32)

dt = checkpoint_config['scenario']['dt']

F = torch.as_tensor(motion_model.F(dummy_state, dt), dtype=torch.float32)
H = torch.as_tensor(measurement_model.H(dummy_state), dtype=torch.float32)
Q = torch.as_tensor(motion_model.Q(dt), dtype=torch.float32)

P0_fixed = torch.as_tensor(config["initialization"]["P0"], dtype=torch.float32)
x0 = torch.zeros(4, dtype=torch.float32)

# Define rollout covariance model
sigma_scale = config['covariance']['sigma_scale']
min_diag = config['covariance']['min_diag']

rollout_cov_model = RolloutCovarianceModel(denoiser=denoiser, sigma_scale=sigma_scale, min_diag=min_diag, F=F, H=H, Q=Q, dt=dt)

# Rollout NLL loss function
def nll_loss_fn(output, targets):
    x_filtered, P_filtered = output

    pos_filtered = x_filtered[:, :, :2] # Only take position components for NLL loss
    P_filtered = P_filtered[:, :, :2, :2] # Only take position components
    jitter = 1e-6 * torch.eye(2, device=P_filtered.device).unsqueeze(0).unsqueeze(0) # Add small jitter for numerical stability
    P_filtered_jitter = P_filtered + jitter

    L_pos = torch.linalg.cholesky(P_filtered_jitter)

    gaussian_nll_loss = gaussian_nll(pos_filtered, targets, L_pos)

    return gaussian_nll_loss

# Refresh warm states cache for the dataset
def refresh_warm_cache(dataset: RolloutWindowDataset, x0_init: torch.Tensor, P0_init: torch.Tensor) -> None:
    
    cache_x, cache_P = compute_warm_states(rollout_cov_model, dataset.windows, mean, std, x0_init, P0_init)
    
    dataset.cache_x = cache_x
    dataset.cache_P = cache_P

# Refresh warm states cache for both train and val datasets
def refresh_all_caches() -> None:
    print("Refreshing warm states")
    refresh_warm_cache(train_dataset, x0_init_train, P0_init_train)
    refresh_warm_cache(val_dataset, x0_init_val, P0_init_val)

# Compute warm states for each trajectory in the train dataset
n_traj, n_windows_per_traj, _, _ = train_dataset.windows.shape
first_windows = train_dataset.windows[:, 0]

first_windows = (first_windows - mean) / std
first_windows = torch.from_numpy(first_windows).float()

with torch.no_grad():
    first_pos = rollout_cov_model.adaptive_denoiser.denoiser(first_windows)

    zero_velocity = torch.zeros((n_traj, 2), dtype=torch.float32)
    x0_init_train = torch.cat([first_pos, zero_velocity], dim=1)

    P0_init_train = P0_fixed.expand(n_traj, -1, -1)

    refresh_warm_cache(train_dataset, x0_init_train, P0_init_train)


# Compute warm states for each trajectory in the val dataset
n_traj, n_windows_per_traj, _, _ = val_dataset.windows.shape
first_windows = val_dataset.windows[:, 0]

first_windows = (first_windows - mean) / std
first_windows = torch.from_numpy(first_windows).float()

with torch.no_grad():
    first_pos = rollout_cov_model.adaptive_denoiser.denoiser(first_windows)

    zero_velocity = torch.zeros((n_traj, 2), dtype=torch.float32)
    x0_init_val = torch.cat([first_pos, zero_velocity], dim=1)

    P0_init_val = P0_fixed.expand(n_traj, -1, -1)

    refresh_warm_cache(val_dataset, x0_init_val, P0_init_val)


# Save history after training/evaluating the model
history = train(rollout_cov_model, train_loader, val_loader, config, nll_loss_fn, on_epoch_start=refresh_all_caches)

# Save this to processed data
checkpoint_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "rollout_covariance_checkpoint.pt"
save_checkpoint(path=checkpoint_path, 
                model=rollout_cov_model, 
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
plt.savefig("figures/train_rollout_covariance.png")
plt.show()

