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
from fusion.models.adaptive_position import AdaptivePosition2D
from fusion.sim.noise import contaminated_gaussian_noise
from fusion.learned.checkpoints import load_checkpoint
from fusion.learned.datasets import load_denoiser_dataset
from fusion.learned.denoiser import build_denoiser
from fusion.learned.covariance import build_adaptive_denoiser, build_rollout_covariance_model, cov_from_chol

# This script will run a comparison for six different conditions (same measurements except for noise)
# 1) EKF with contaminated noise; This is the floor for performance and should be the weakest
# 2) Denoiser alone; This isolates what work is being done by the denoiser and helps test whether the filter adds any value
# 3) Denoiser fed to KF; If the denoiser is able to clean the noise, it should allow the KF to perform closer to its clean noise ceiling
# 4) Denoiser fed to KF with R scaled by best_alpha (fixed, hand-tuned measurement noise)
# 5) Denoiser fed to KF with R predicted per-window by the adaptive (learned covariance) model
# 6) EKF with clean noise; This is the ceiling for performance and theoretically the best outcome
 
# UKF is not used in this script as it produce almost identical results as EKF (as seen in stage 7; demo_contaminated_noise.py)
# EOIR is not included deliberately to reduce the scope of this comparison
# Given that Denoiser produces Cartesian positions, it currently cannot be fed into EKF (which has radar/bearing as input), so KF is used

# Conditions to be used in plots (labels, NEES present)
conditions = {
    "ekf_cont": ("EKF (Contaminated)", True),
    "denoiser": ("Denoiser Alone", False),  # NEES not suitable for Denoiser
    "kf": ("Denoiser to KF", True),
    "kf_alpha": ("Denoiser to KF & Alpha",True),
    "kf_adaptive": ("Denoiser to KF & Adaptive R", True),
    "kf_rollout": ("Denoiser to KF & Rollout R", True),
    "ekf_clean":("EKF (Clean)",True),
}

# More streamlined results aggregation
def make_results() -> dict[str, dict[str, list]]:
    """
    Creates a new results dicitonary when called
    """
    return {
        name: {"pos": [], **({"nees": []} if has_nees else {})} for name, (_, has_nees) in conditions.items()
    }

# Load denoiser and adaptive denoiser models, norm stats, and config data from prior created dataset (denoiser_checkpoint)
model, normalization_stats, checkpoint_config = load_checkpoint(path="data/processed/denoiser_checkpoint.pt", build_model_fn=build_denoiser)
model.eval()

rollout_model, _, _ = load_checkpoint(path="data/processed/rollout_covariance_checkpoint.pt", build_model_fn=build_rollout_covariance_model)
rollout_model.eval()

adaptive_model, _, _ = load_checkpoint(path="data/processed/adaptive_r_checkpoint.pt", build_model_fn=build_adaptive_denoiser)
adaptive_model.eval()

# Extract config variables
mc_trials = config['comparison']['mc_trials']
batch_size = config['training']['batch_size']
best_alpha = config['baseline']['best_alpha']

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
linear_model_alpha = LinearPosition2D(sigma_pos=best_alpha*sigma_pos)

# To collect data from multiple trials
def run_trial(seed) -> tuple[dict[str, dict[str, list]], list[np.ndarray]]:
    rng = np.random.default_rng(seed)  # For reproducibility

    # Adaptive Position Model is moved to inside to ensure it is fresh at each trial
    adaptive_position_model = AdaptivePosition2D(R_init=np.eye(dim_pos)) # Initial R gets overwritten
    rollout_position_model = AdaptivePosition2D(R_init=np.eye(dim_pos)) # Initial R gets overwritten

    # Fresh filters for each trial (R is carried, but is overwritten)
    filters = {
        "ekf_cont":    ExtendedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0),
        "kf":          KalmanFilter(motion=motion_model, measurement=linear_model, x0=x0_filter, P0=P0),
        "kf_alpha":    KalmanFilter(motion=motion_model, measurement=linear_model_alpha, x0=x0_filter, P0=P0),
        "kf_adaptive": KalmanFilter(motion=motion_model, measurement=adaptive_position_model, x0=x0_filter, P0=P0),
        "kf_rollout":  KalmanFilter(motion=motion_model, measurement=rollout_position_model, x0=x0_filter, P0=P0),
        "ekf_clean":   ExtendedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0),
    }

    # Set up arrays
    results = make_results()
    ground_truth = []
    z_contaminated_history = []
    
    # Learned-model outputs (None until the window is full)
    denoiser_z = None
    adaptive_z = None
    rollout_z = None

  
    for i in range(1, n_steps):
        truth = trajectory[i]
        ground_truth.append(truth[:2])

        # Generate clean and contaminated measurements
        z_clean = radar_model.h(truth) + rng.normal(0, [sigma_range, sigma_bearing])
        z_contaminated = radar_model.h(truth) + contaminated_gaussian_noise(rng, radar_model.R(), epsilon=epsilon, kappa=kappa)
        z_contaminated_history.append(z_contaminated)

        # Once enough steps for window, feed current window to denoiser and adaptive model
        if i >= window:
            curr_window = np.array(z_contaminated_history[-window:])
            normalized_window = (curr_window - normalization_stats['mean']) / normalization_stats['std']
            # Make tensor with batch dimension
            window_tensor = torch.from_numpy(normalized_window).float().unsqueeze(0)

            with torch.no_grad():
                denoiser_z = model(window_tensor).squeeze(0).numpy()

                # Adaptive model position and Cholesky to R
                adaptive_pos, L = adaptive_model(window_tensor)
                adaptive_z = adaptive_pos.squeeze(0).numpy()
                adaptive_position_model.set_R(cov_from_chol(L).squeeze(0).numpy())

                # Rollout model position and Cholesky to R
                rollout_pos, L_rollout = rollout_model.adaptive_denoiser(window_tensor)
                rollout_z = rollout_pos.squeeze(0).numpy()
                rollout_position_model.set_R(cov_from_chol(L_rollout).squeeze(0).numpy())

        # Measurements fed to each filter this step (prior to update)
        measurements = {
            "ekf_cont": z_contaminated,
            "kf": denoiser_z,
            "kf_alpha": denoiser_z,
            "kf_adaptive": adaptive_z,
            "kf_rollout": rollout_z,
            "ekf_clean": z_clean,
        }

        # Predict, update, and record for every filter
        for name, filt in filters.items():
            filt.predict(dt)
            if measurements[name] is not None:
                filt.update(measurements[name])
            results[name]["pos"].append(filt.x[:2].copy())
            results[name]["nees"].append(nees(filt.x, filt.P, truth))

        # Denoiser alone only had raw measurement and is NaN until window fills up
        results["denoiser"]["pos"].append(denoiser_z if denoiser_z is not None else np.full(dim_pos, np.nan))

    return results, ground_truth

all_results = make_results()
all_ground_truth = []

# Run the simulation
for trial in range(mc_trials):
    results, ground_truth = run_trial(trial)
 
    for name, cond in results.items():
        all_results[name]["pos"].append(cond["pos"])
        if "nees" in cond:
            all_results[name]["nees"].append(cond["nees"])
 
    all_ground_truth.append(ground_truth)

# Given kf and denoisers only have data after window iterations slice to window-1
all_ground_truth = np.array(all_ground_truth)[:, window-1:].reshape(-1, 2)
 
for name, cond in all_results.items():
    label = conditions[name][0]

    # Calculate RMSE
    pos = np.array(cond["pos"])[:, window-1:].reshape(-1, 2)
    cond["rmse"] = position_rmse(pos, all_ground_truth)
    print(f"Root Mean Square Error ({label}): {cond['rmse']}")

    # Calculate mean NEES when relevant
    if "nees" in cond:
        cond["mean_nees"] = np.mean(np.array(cond["nees"])[:, window-1:])
 
print()
for name, cond in all_results.items():
    if "mean_nees" in cond:
        print(f"Average NEES ({conditions[name][0]}): {cond['mean_nees']}")

# Plot RMSE across all scenarios
names_rmse = [n for n, c in all_results.items() if "rmse" in c]
labels_1 = [conditions[n][0] for n in names_rmse]
values_1 = [all_results[n]["rmse"] for n in names_rmse]
plt.figure(figsize=(12, 6))
plt.bar(labels_1, values_1)
plt.xlabel("Scenarios")
plt.ylabel("RMSE")
plt.title("RMSE for Each Scenario")
plt.xticks(rotation=20, ha="right")
plt.grid()
plt.tight_layout()
plt.savefig("figures/run_comparison_1_2.png")
plt.show()
 
# Plot NEES across applicable conditions
names_nees = [n for n, c in all_results.items() if "mean_nees" in c]
labels_2 = [conditions[n][0] for n in names_nees]
values_2 = [all_results[n]["mean_nees"] for n in names_nees]
plt.figure(figsize=(12, 6))
plt.bar(labels_2, values_2)
plt.axhline(y=4, linestyle=':', label="Ideal NEES (state dim)")
plt.xlabel("Scenarios")
plt.ylabel("NEES")
plt.title("NEES for Each Scenario")
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/run_comparison_2_2.png")
plt.show()
 


