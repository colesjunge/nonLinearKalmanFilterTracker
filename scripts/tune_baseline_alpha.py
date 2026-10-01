from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
import torch
from scipy.optimize import brentq

from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D
from tracking.filters.kalman import KalmanFilter
from tracking.eval.metrics import nees

from fusion.models.radar import RadarRangeBearing
from fusion.sim.noise import contaminated_gaussian_noise
from fusion.learned.checkpoints import load_checkpoint
from fusion.learned.datasets import load_denoiser_dataset
from fusion.learned.denoiser import build_denoiser

# This script is to determine alpha for the other scripts

# Load denoiser model, norm stats, and config data from prior created dataset (denoiser_checkpoint)
model, normalization_stats, checkpoint_config = load_checkpoint(path="data/processed/denoiser_checkpoint.pt", build_model_fn=build_denoiser)
model.eval()

# Extract config variables
batch_size = config['training']['batch_size']

sigma_range = checkpoint_config['scenario']['sigma_range']
sigma_bearing = checkpoint_config['scenario']['sigma_bearing']
epsilon = checkpoint_config['scenario']['epsilon']
kappa = checkpoint_config['scenario']['kappa']
window = checkpoint_config['dataset']['window']
dt = checkpoint_config['scenario']['dt']
n_steps = checkpoint_config['scenario']['n_steps']
val_fraction = checkpoint_config['dataset']['val_fraction']
dim_pos = checkpoint_config['model']['dim_pos']
checkpoint_seed = checkpoint_config['seed']

alpha_grid = config['baseline']['alpha_grid']
tuning_seeds = config['baseline']['tuning_seeds']

# Compute sigma_pos empirically
_, val_dataset, _ = load_denoiser_dataset(path='data/processed/denoiser_dataset.npz', val_fraction=val_fraction, seed=checkpoint_seed)

val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Shuffling unnecessary for val set

errors = []
with torch.no_grad():
    for windows, targets in val_loader:
        predictions = model(windows) # Forward pass
        errors.append(predictions-targets) 

# Error is (batch_size, dim_pos) to become (n_val_samples, dim_pos)
all_errors = torch.cat(errors, dim=0)

# This is done as the original RMSE function would inflate bias if present
std = all_errors.std(dim=0)
sigma_pos = torch.sqrt(torch.mean(std ** 2)).item()

# True target and trajectory
x0_true = np.array([-50.0, -25.0, 0.0, 5.0]) # px, py, vx, vy

# Ground truth trajectory
trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

x0_filter = np.array([-45.0, -26.0, -1.0, 4.5]) # Initial state guess (random numbers I've chosen)
P0 = np.array([[900, 0, 0, 0], 
                [0, 4, 0, 0], 
                [0, 0, 4, 0], 
                [0, 0, 0, 4]]) # Initial uncertainty (random numbers I've chosen)

radar_model = RadarRangeBearing(sigma_range=sigma_range, sigma_bearing=sigma_bearing)
motion_model = ConstantVelocity2D(sigma_a=0.0)


# To collect data from multiple trials
def run_trial(seed, linear_model) -> tuple[list[np.ndarray], list[float], list[np.ndarray]]:

    rng = np.random.default_rng(seed)  # For reproducibility
    # KF
    kf = KalmanFilter(motion=motion_model, measurement=linear_model, x0=x0_filter, P0=P0)

    ground_truth = []
    filtered_estimates_kf = []
    nees_kf = []
    z_contaminated_history = []


    for i in range(1, n_steps):
        ground_truth.append(trajectory[i][:2])

        # Generate clean and contaminated measurements
        z_clean = radar_model.h(trajectory[i]) + rng.normal(0, [sigma_range, sigma_bearing])
        z_contaminated = radar_model.h(trajectory[i]) + contaminated_gaussian_noise(rng, radar_model.R(), epsilon=epsilon, kappa=kappa)
        z_contaminated_history.append(z_contaminated)

        # Once enough steps for window, feed current window to denoiser
        if i >= window:
            curr_window = z_contaminated_history[-window:]
            curr_window = np.array(curr_window)

            normalized_window = (curr_window - normalization_stats['mean']) / normalization_stats['std']
            #Make tensor with batch dimension
            window_tensor = torch.from_numpy(normalized_window).float().unsqueeze(0)

            with torch.no_grad():
                prediction = model(window_tensor)

                denoiser_z = prediction.squeeze(0).numpy()

        # Predict and update

        kf.predict(dt)
        if i >= window:
            kf.update(denoiser_z)

        filtered_estimates_kf.append(kf.x[:2])
        nees_kf.append(nees(kf.x, kf.P, trajectory[i]))


    return filtered_estimates_kf, nees_kf, ground_truth

# Calculate mean NEES for each alpha
def mean_nees_for_alpha(alpha) -> float:

    # Build measurement model for this alpha
    linear_model = LinearPosition2D(sigma_pos=alpha * sigma_pos)

    # Accumulate NEES across all tuning seeds
    alpha_nees = []

    for seed in tuning_seeds:
        _, nees_kf, _ = run_trial(seed, linear_model)

        alpha_nees.append(nees_kf)

    # Convert to array
    alpha_nees = np.array(alpha_nees)

    # Remove iterations before the denoiser starts
    alpha_nees = alpha_nees[:, window-1:]

    # Average over seeds
    mean_nees = np.mean(alpha_nees)

    return mean_nees

alpha_results = []

for alpha in alpha_grid:
    mean_nees = mean_nees_for_alpha(alpha)

    alpha_results.append((alpha, mean_nees))

    print(
        f"alpha = {alpha:.4f}, "
        f"mean NEES = {mean_nees:.4f}"
    )
# Find the alpha whose mean NEES is closest to the target value of 4 (from prior trial, it should be around 4)
best_alpha = brentq(lambda alpha: mean_nees_for_alpha(alpha) - 4, a=2.0,b=5.0) # 3.8763

best_mean_nees = mean_nees_for_alpha(best_alpha)
print("Alpha tuning results:")
for alpha, mean_nees in alpha_results:
    print(f"alpha = {alpha:.4f}, mean NEES = {mean_nees:.4f}")

print(f"Best alpha = {best_alpha:.4f}")
print(f"mean NEES = {best_mean_nees:.4f}")






