from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import torch
import numpy as np
import matplotlib.pyplot as plt


from fusion.learned.denoiser import Denoiser
from fusion.learned.dynamics import ResidualDynamicsNet
from fusion.learned.full_pipeline import FullPipelineModel
from fusion.learned.covariance import gaussian_nll
from fusion.learned.datasets import FullPipelineRolloutDataset, load_full_pipeline_dataset, compute_full_pipeline_warm_states
from fusion.learned.train import train
from fusion.learned.checkpoints import save_checkpoint
from tracking.models.measurement import LinearPosition2D
from tracking.models.motion import ConstantVelocity2D

# Set up config variables
seed = config["seed"]
dt = config["scenario"]["dt"]

window = config["dataset"]["window"]
val_fraction = config["dataset"]["val_fraction"]

K = config["rollout"]["K"]

hidden_channels = config["model"]["hidden_channels"]
n_layers = config["model"]["n_layers"]
kernel_size = config["model"]["kernel_size"]
dim_z = config["model"]["dim_z"]
dim_pos = config["model"]["dim_pos"]

hidden_dim = config["dynamics_net"]["hidden_dim"]
sigma_scale_dn = config["dynamics_net"]["sigma_scale"]
min_diag_dn = config["dynamics_net"]["min_diag"]

sigma_scale_cov = config["covariance"]["sigma_scale"]
min_diag_cov = config["covariance"]["min_diag"]

P0_bearing = config["initialization"]["P0"]

batch_size = config["training"]["batch_size"]
epochs = config["training"]["epochs"]
learning_rate = config["training"]["learning_rate"]
grad_clip_norm = config["training"]["grad_clip_norm"]

dtype = torch.float32

# Set seed explicitly
torch.manual_seed(seed)

# Set up dummy state and F/H
dummy_x = np.zeros(4, dtype=np.float32)
F = torch.as_tensor(ConstantVelocity2D(sigma_a=0.0).F(dummy_x, dt), dtype=dtype)
H = torch.as_tensor(LinearPosition2D(sigma_pos=1.0).H(dummy_x), dtype=dtype)

# Set up denoiser
denoiser = Denoiser(dim_z=dim_z,
                    dim_pos=dim_pos,
                    hidden_channels=hidden_channels,
                    n_layers=n_layers,
                    kernel_size=kernel_size)


# Build ResidualDynamicsNet
dynamics_net = ResidualDynamicsNet(dim_x=4, 
                                   hidden_dim=hidden_dim,
                                   sigma_scale=sigma_scale_dn, 
                                   min_diag=min_diag_dn)

# Build FullPipelineModel (sigma_scale and min_diag come from covariance not dynamics_net)

full_pipeline_model = FullPipelineModel(denoiser=denoiser,
                                        sigma_scale=sigma_scale_cov,
                                        min_diag=min_diag_cov,
                                        dynamics_net=dynamics_net,
                                        F=F,
                                        H=H,
                                        dt=dt)

# Get train_dataset, val_dataset, and stats_dict
train_dataset, val_dataset, stats_dict = load_full_pipeline_dataset(path='data/processed/full_pipeline_dataset.npz', 
                                                                    K=K, 
                                                                    val_fraction=val_fraction, 
                                                                    seed=seed, 
                                                                    window=window)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Shuffling unnessecary

# Training set x0, P0, h0
n_traj_train = train_dataset.windows.shape[0]
zero_velocity_train = torch.zeros(n_traj_train, 2)

first_meas_train = train_dataset.windows[:, 0, -1] # (n_traj, 2) (Range, Bearing)
px_train = first_meas_train[:, 0] * np.cos(first_meas_train[:, 1])
py_train = first_meas_train[:, 0] * np.sin(first_meas_train[:, 1])

px_train = torch.from_numpy(px_train).float().unsqueeze(1) # (n_traj, 1)
py_train = torch.from_numpy(py_train).float().unsqueeze(1) # (n_traj, 1)

x0_init_train = torch.cat([px_train, py_train, zero_velocity_train], dim=1)
P0_init_train = torch.as_tensor(P0_bearing, dtype=dtype).expand(n_traj_train, -1, -1)
h0_init_train = torch.zeros(n_traj_train, hidden_dim)

# Val set x0, P0, h0
n_traj_val= val_dataset.windows.shape[0]
zero_velocity_val = torch.zeros(n_traj_val, 2)

first_meas_val = val_dataset.windows[:, 0, -1] # (n_traj, 2) (Range, Bearing)
px_val = first_meas_val[:, 0] * np.cos(first_meas_val[:, 1])
py_val = first_meas_val[:, 0] * np.sin(first_meas_val[:, 1])

px_val = torch.from_numpy(px_val).float().unsqueeze(1) # (n_traj, 1)
py_val = torch.from_numpy(py_val).float().unsqueeze(1) # (n_traj, 1)

x0_init_val = torch.cat([px_val, py_val, zero_velocity_val], dim=1)
P0_init_val = torch.as_tensor(P0_bearing, dtype=dtype).expand(n_traj_val, -1, -1)
h0_init_val = torch.zeros(n_traj_val, hidden_dim)

# Refresh warm caches
def refresh_warm_cache(dataset: FullPipelineRolloutDataset, mean, std, x0_init, P0_init, h0_init) -> None:
    # Gradients frozen within compute_full_pipeline_warm_states
    cache_x, cache_P, cache_h = compute_full_pipeline_warm_states(full_pipeline_model, 
                                                                  dataset.windows, 
                                                                  mean, 
                                                                  std, 
                                                                  x0_init, 
                                                                  P0_init, 
                                                                  h0_init)
    dataset.cache_x = cache_x
    dataset.cache_P = cache_P
    dataset.cache_h = cache_h

def refresh_all_caches() -> None:
    print("Refreshing warm states")
    refresh_warm_cache(train_dataset, stats_dict["mean"], stats_dict["std"], x0_init_train, P0_init_train, h0_init_train)
    refresh_warm_cache(val_dataset, stats_dict["mean"], stats_dict["std"], x0_init_val, P0_init_val, h0_init_val)

# Populate both caches once before training starts
refresh_all_caches()

# Rollout NLL loss function (pos and vel)
def nll_loss_fn(output, targets):
    x_filtered, P_filtered, h_filtered = output

    jitter = 1e-6 * torch.eye(4, device=P_filtered.device).unsqueeze(0).unsqueeze(0) # Add small jitter for numerical stability
    P_filtered_jitter = P_filtered + jitter

    L = torch.linalg.cholesky(P_filtered_jitter)

    gaussian_nll_loss = gaussian_nll(x_filtered, targets, L)

    return gaussian_nll_loss

# Save history after training/evaluating the model
history = train(full_pipeline_model, train_loader, val_loader, config, nll_loss_fn, on_epoch_start=refresh_all_caches)

# Save this to processed data
checkpoint_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "full_pipeline_checkpoint.pt"
save_checkpoint(path=checkpoint_path, 
                model=full_pipeline_model, 
                normalization_stats=stats_dict, 
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
plt.savefig("figures/train_full_pipeline.png")
plt.show()


