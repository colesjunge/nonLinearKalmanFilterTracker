from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
import torch
import matplotlib.pyplot as plt

from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D
from tracking.filters.kalman import KalmanFilter
from tracking.eval.metrics import position_rmse, nees

from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.models.radar import RadarRangeBearing
from fusion.sim.noise import contaminated_gaussian_noise
from fusion.learned.checkpoints import load_checkpoint
from fusion.learned.datasets import load_denoiser_dataset

# This script will run a comparison for four different conditions (same measurements except for noise)
# 1) EKF with contaminated noise; This is the floor for performance and should be the weakest
# 2) Denoiser alone; This isolates what work is being done by the denoiser and helps test whether the filter adds any value
# 3) Denoiser fed to KF; If the denoiser is able to clean the noise, it should allow the KF to perform closer to its clean noise ceiling
# 4) EKF with clean noise; This is the ceiling for performance and theoretically the best outcome

# UKF is not used in this script as it produce almost identical results as EKF (as seen in stage 7; demo_contaminated_noise.py)
# EOIR is not included deliberately to reduce the scope of this comparison
# Given that Denoiser produces Cartesian positions, it currently cannot be fed into EKF (which has radar/bearing as input), so KF is used


# Load denoiser model, norm stats, and config data from prior created dataset (denoiser_checkpoint)
model, normalization_stats, checkpoint_config = load_checkpoint(path="data/processed/denoiser_checkpoint.pt")
model.eval()

# Extract config variables
mc_trials = config['comparison']['mc_trials']
batch_size = config['training']['batch_size']

sigma_range = checkpoint_config['scenario']['sigma_range']
sigma_bearing = checkpoint_config['scenario']['sigma_bearing']
epsilon = checkpoint_config['scenario']['epsilon']
kappa = checkpoint_config['scenario']['kappa']
window = checkpoint_config['dataset']['window']
dt = checkpoint_config['scenario']['dt']
n_steps = checkpoint_config['scenario']['n_steps']
val_fraction = checkpoint_config['dataset']['val_fraction']
seed = checkpoint_config['seed']
dim_pos = checkpoint_config['model']['dim_pos']

# Compute sigma_pos empirically
_, val_dataset, _ = load_denoiser_dataset(path='data/processed/denoiser_dataset.npz', val_fraction=val_fraction, seed=seed)

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
linear_model = LinearPosition2D(sigma_pos=sigma_pos)

# To collect data from multiple trials
def run_trial(seed) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float], list[float], list[float], list[np.ndarray]]:
    rng = np.random.default_rng(seed)  # For reproducibility
    # Conditions 1,3,4 (2 absent as just denoiser)
    ekf1= ExtendedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)
    kf = KalmanFilter(motion=motion_model, measurement=linear_model, x0=x0_filter, P0=P0)
    ekf4 = ExtendedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)

    ground_truth = []
    filtered_estimates_ekf1 = []
    filtered_estimates_denoiser = []
    filtered_estimates_kf = []
    filtered_estimates_ekf4 = []
    nees_ekf1 = []
    # NEES not suitable for Denoiser
    nees_kf = []
    nees_ekf4 = []
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
        ekf1.predict(dt)
        ekf1.update(z_contaminated)

        kf.predict(dt)
        if i >= window:
            kf.update(denoiser_z)
        ekf4.predict(dt)
        ekf4.update(z_clean)


        filtered_estimates_ekf1.append(ekf1.x[:2])
        nees_ekf1.append(nees(ekf1.x, ekf1.P, trajectory[i]))

        if i >= window:
            filtered_estimates_denoiser.append(denoiser_z)
        else:
            filtered_estimates_denoiser.append(np.full(dim_pos, np.nan))

        filtered_estimates_kf.append(kf.x[:2])
        nees_kf.append(nees(kf.x, kf.P, trajectory[i]))

        filtered_estimates_ekf4.append(ekf4.x[:2])
        nees_ekf4.append(nees(ekf4.x, ekf4.P, trajectory[i]))

    return filtered_estimates_ekf1, filtered_estimates_denoiser, filtered_estimates_kf, filtered_estimates_ekf4, nees_ekf1, nees_kf, nees_ekf4, ground_truth

all_filtered_estimates_ekf1 = []
all_filtered_estimates_denoiser = []
all_filtered_estimates_kf= []
all_filtered_estimates_ekf4 = []
all_nees_ekf1 = []
all_nees_kf = []
all_nees_ekf4= []
all_ground_truth = []

# Run the simulation
for trial in range(mc_trials): 
    filtered_estimates_ekf1, filtered_estimates_denoiser, filtered_estimates_kf, filtered_estimates_ekf4, nees_ekf1, nees_kf, nees_ekf4, ground_truth = run_trial(trial)

    all_filtered_estimates_ekf1.append(filtered_estimates_ekf1)
    all_filtered_estimates_denoiser.append(filtered_estimates_denoiser)
    all_filtered_estimates_kf.append(filtered_estimates_kf)
    all_filtered_estimates_ekf4.append(filtered_estimates_ekf4)
    all_nees_ekf1.append(nees_ekf1)
    all_nees_kf.append(nees_kf)
    all_nees_ekf4.append(nees_ekf4)
    all_ground_truth.append(ground_truth)

# Given kf and denoiser only have data after window iterations slice to window-1
all_filtered_estimates_ekf1 = np.array(all_filtered_estimates_ekf1)[:, window-1:].reshape(-1, 2)
all_filtered_estimates_denoiser = np.array(all_filtered_estimates_denoiser)[:, window-1:].reshape(-1, 2)
all_filtered_estimates_kf = np.array(all_filtered_estimates_kf)[:, window-1:].reshape(-1, 2)
all_filtered_estimates_ekf4 = np.array(all_filtered_estimates_ekf4)[:, window-1:].reshape(-1, 2)
all_nees_ekf1 = np.array(all_nees_ekf1)[:, window-1:]
all_nees_kf = np.array(all_nees_kf)[:, window-1:]
all_nees_ekf4 = np.array(all_nees_ekf4)[:, window-1:]
all_ground_truth = np.array(all_ground_truth)[:, window-1:].reshape(-1, 2)

# Calculate mean NEES for 1,3,4
mean_ekf1 = np.mean(all_nees_ekf1)
mean_kf = np.mean(all_nees_kf)
mean_ekf4= np.mean(all_nees_ekf4)


# Calculate RMSE between ground truth and filtered estimates
filtered_rmse_ekf1 = position_rmse(all_filtered_estimates_ekf1, all_ground_truth)
print(f"Root Mean Square Error (EKF Filtered Estimates with Contamination): {filtered_rmse_ekf1}")

filtered_rmse_denoiser = position_rmse(all_filtered_estimates_denoiser, all_ground_truth)
print(f"Root Mean Square Error (Denoiser Estimates): {filtered_rmse_denoiser}")

filtered_rmse_kf = position_rmse(all_filtered_estimates_kf, all_ground_truth)
print(f"Root Mean Square Error (KF Filtered Estimates with Denoiser): {filtered_rmse_kf}")

filtered_rmse_ekf4 = position_rmse(all_filtered_estimates_ekf4, all_ground_truth)
print(f"Root Mean Square Error (EKF Filtered Estimates with no Contamination): {filtered_rmse_ekf4}")

# Calculate Average NEES for applicable scenarios
print(f"Average NEES (EKF with Contaminated Noise): {mean_ekf1}")
print(f"Average NEES (KF with Denoiser): {mean_kf}")
print(f"Average NEES (EKF with Clean Noise): {mean_ekf4}")

# Plot RMSE across all four scenarios
labels_1 = ["EKF (Contaminated)", "Denoiser Alone", "Denoiser to KF", "EKF (Clean)"]
values_1 = [filtered_rmse_ekf1, filtered_rmse_denoiser, filtered_rmse_kf, filtered_rmse_ekf4]
plt.figure(figsize=(10, 6))
plt.bar(labels_1, values_1)
plt.xlabel("Scenarios")
plt.ylabel("RMSE")
plt.title("RMSE for Each Scenario")
plt.grid()
plt.tight_layout()
plt.savefig("figures/run_comparison_1.png")
plt.show()

#Plot NEES across three applicable conditions
labels_2 = ["EKF (Contaminated)", "Denoiser to KF", "EKF (Clean)"]
values_2 = [mean_ekf1, mean_kf, mean_ekf4]
plt.figure(figsize=(10, 6))
plt.bar(labels_2, values_2)
plt.axhline(y=4, linestyle=':', label="Ideal NEES (state dim)")
plt.xlabel("Scenarios")
plt.ylabel("NEES")
plt.title("NEES for Each Scenario")
plt.grid()
plt.tight_layout()
plt.savefig("figures/run_comparison_2.png")
plt.show()



