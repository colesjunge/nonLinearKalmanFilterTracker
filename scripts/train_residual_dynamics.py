from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import torch
import torch.nn.functional as nnf
import numpy as np
from fusion.learned.dynamics import ResidualDynamicsNet
from fusion.learned.residual_dynamics_kf import ResidualDynamicsKF


from fusion.learned.datasets import ManeuverRolloutDataset, load_maneuver_dataset, compute_maneuver_warm_states
from fusion.learned.train import train
from fusion.learned.checkpoints import save_checkpoint
from tracking.models.measurement import LinearPosition2D
from tracking.models.motion import ConstantVelocity2D
import matplotlib.pyplot as plt


# Set seed explicitly
torch.manual_seed(config['seed'])

K = config['rollout']['K']

# Load train/validation data from prior created dataset
train_dataset, val_dataset = load_maneuver_dataset(
           path='data/processed/maneuver_dataset.npz', 
           K=K,
           val_fraction=config['maneuver']['val_fraction'], seed=config['seed'],
           sigma_pos=config['maneuver']['sigma_pos'])

# Wrap dataset
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False) # Shuffling unnecessary for val set

# Build F, H, R (Q comes from dynamics_net) matrices for differentiable KF
motion_model = ConstantVelocity2D(sigma_a=0.0)
measurement_model = LinearPosition2D(sigma_pos=config['maneuver']['sigma_pos'])

dummy_state = np.zeros(4, dtype=np.float32)

dt = config['scenario']['dt']

F = torch.as_tensor(motion_model.F(dummy_state, dt), dtype=torch.float32)
H = torch.as_tensor(measurement_model.H(dummy_state), dtype=torch.float32)
R = torch.as_tensor(measurement_model.R(), dtype=torch.float32)

P0_fixed = torch.as_tensor(config["initialization"]["P0"], dtype=torch.float32)
x0 = torch.zeros(4, dtype=torch.float32)

# Define residual dynamics model
dynamics_net = ResidualDynamicsNet(dim_x=4, 
                                   hidden_dim=config['dynamics_net']['hidden_dim'],
                                   sigma_scale=config['dynamics_net']['sigma_scale'], 
                                   min_diag=config['dynamics_net']['min_diag'])
model = ResidualDynamicsKF(dynamics_net, F, H, R, dt=dt)


# MSE loss function (all four dimensions)
def mse_loss_fn(output, targets):
    x_seq, P_seq, h_seq = output
    return nnf.mse_loss(x_seq, targets)

# Refresh warm states cache for the dataset
def refresh_warm_cache(dataset: ManeuverRolloutDataset, x0_init: torch.Tensor, P0_init: torch.Tensor, h0_init: torch.Tensor) -> None:
    
    cache_x, cache_P, cache_h = compute_maneuver_warm_states(model, z=dataset.z, x0_init=x0_init, P0_init=P0_init, h0_init=h0_init)
    
    dataset.cache_x = cache_x
    dataset.cache_P = cache_P
    dataset.cache_h = cache_h

# Refresh warm states cache for both train and val datasets
def refresh_all_caches() -> None:
    print("Refreshing warm states")
    refresh_warm_cache(train_dataset, x0_init_train, P0_init_train, h0_init_train)
    refresh_warm_cache(val_dataset, x0_init_val, P0_init_val, h0_init_val)

# Train dataset initial warm-start inputs
n_traj = train_dataset.z.shape[0]
first_pos_train = torch.from_numpy(train_dataset.z[:, 0]).float()
zero_velocity_train = torch.zeros(n_traj, 2) # First instance does not have enough info for velocity so set to 0
x0_init_train = torch.cat([first_pos_train, zero_velocity_train], dim=1)
P0_init_train = P0_fixed.expand(n_traj, -1, -1)
h0_init_train = torch.zeros(n_traj, config['dynamics_net']['hidden_dim'])

# Val dataset initial warm-start inputs
n_traj_val = val_dataset.z.shape[0]
first_pos_val = torch.from_numpy(val_dataset.z[:, 0]).float()
zero_velocity_val = torch.zeros(n_traj_val, 2) # First instance does not have enough info for velocity so set to 0
x0_init_val = torch.cat([first_pos_val, zero_velocity_val], dim=1)
P0_init_val = P0_fixed.expand(n_traj_val, -1, -1)
h0_init_val = torch.zeros(n_traj_val, config['dynamics_net']['hidden_dim'])

# Populate both caches once before training starts
refresh_all_caches()


# Save history after training/evaluating the model
history = train(model, train_loader, val_loader, config, mse_loss_fn, on_epoch_start=refresh_all_caches)

# Save this to processed data
checkpoint_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "residual_dynamics_checkpoint.pt"
save_checkpoint(path=checkpoint_path, 
                model=model, 
                normalization_stats={}, 
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
plt.savefig("figures/train_residual_dynamics.png")
plt.show()

