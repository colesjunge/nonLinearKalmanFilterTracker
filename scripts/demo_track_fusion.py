import numpy as np
from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.eval.metrics import nees
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.models.radar import RadarRangeBearing
from fusion.models.eoir import EOIRBearingOnly
from fusion.fuse.naive import fuse_naive
from fusion.fuse.covariance_intersection import covariance_intersection
import matplotlib.pyplot as plt

# This script is to compare naive vs covariance intersection 
# This script will only use EKF as the EKF vs UKF comparison is not the focus
# The main analysis will focus on the NEES for each

n_trials = 30

dt = 0.1
n_steps = 100
x0_true = np.array([50.0, 25.0, 0.0, 5.0]) # px, py, vx, vy

# Ground truth trajectory
trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

x0_filter = np.array([45.0, 29.0, -1.0, 4.5]) # Initial state guess (random numbers I've chosen)
P0 = np.array([[10, 0, 0, 0], 
                [0, 13, 0, 0], 
                [0, 0, 4, 0], 
                [0, 0, 0, 4]]) # Initial uncertainty (random numbers I've chosen)

# Filter setup 
motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
sigma_range = 20.0 
sigma_bearing = 0.01
radar_model= RadarRangeBearing(sigma_range=sigma_range, sigma_bearing=sigma_bearing)
eoir_model = EOIRBearingOnly(sigma_bearing=sigma_bearing, sensor_pos=np.array([20, 35]))

# To collect data from multiple trials
def run_trial(seed) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[float], list[float]]:
    rng = np.random.default_rng(seed)  # For reproducibility
    ekf_eoir = ExtendedKalmanFilter(motion_model, eoir_model, x0=x0_filter, P0=P0)
    ekf_radar = ExtendedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)

    filtered_estimates_ekf_fused_naive = []
    filtered_estimates_ekf_fused_ci = []

    filtered_covariances_ekf_fused_naive = []
    filtered_covariances_ekf_fused_ci = []

    nees_naive = []
    nees_ci = []

    for i in range(1, n_steps):

        z_radar = radar_model.h(trajectory[i]) + rng.normal(0, [sigma_range, sigma_bearing])
        z_eoir = eoir_model.h(trajectory[i]) + rng.normal(0, [sigma_bearing])

        # Radar predict/update
        ekf_radar.predict(dt)
        ekf_radar.update(z_radar)

        # EOIR predict/update
        ekf_eoir.predict(dt)
        ekf_eoir.update(z_eoir)

        # Fused updates
        x_naive, P_naive = fuse_naive(ekf_radar.x, ekf_radar.P, ekf_eoir.x, ekf_eoir.P)
        x_ci, P_ci = covariance_intersection(ekf_radar.x, ekf_radar.P, ekf_eoir.x, ekf_eoir.P)

        # Filtered Estimates
        filtered_estimates_ekf_fused_naive.append(x_naive)
        filtered_estimates_ekf_fused_ci.append(x_ci)

        # Filtered Covariances
        filtered_covariances_ekf_fused_naive.append(P_naive.copy())
        filtered_covariances_ekf_fused_ci.append(P_ci.copy())

        nees_naive.append(nees(x_naive, P_naive, trajectory[i]))
        nees_ci.append(nees(x_ci, P_ci, trajectory[i]))

    return filtered_estimates_ekf_fused_naive, filtered_estimates_ekf_fused_ci, filtered_covariances_ekf_fused_naive, filtered_covariances_ekf_fused_ci, nees_naive, nees_ci
    
all_filtered_estimates_ekf_fused_naive = []
all_filtered_estimates_ekf_fused_ci = []
all_filtered_covariances_ekf_fused_naive = []
all_filtered_covariances_ekf_fused_ci = []
all_nees_naive = []
all_nees_ci = []

# Run the simulation
for trial in range(n_trials): 
    filtered_estimates_ekf_fused_naive, filtered_estimates_ekf_fused_ci, filtered_covariances_ekf_fused_naive, filtered_covariances_ekf_fused_ci, nees_naive, nees_ci = run_trial(trial)

    all_filtered_estimates_ekf_fused_naive.append(filtered_estimates_ekf_fused_naive)
    all_filtered_estimates_ekf_fused_ci.append(filtered_estimates_ekf_fused_ci)
    all_filtered_covariances_ekf_fused_naive.append(filtered_covariances_ekf_fused_naive)
    all_filtered_covariances_ekf_fused_ci.append(filtered_covariances_ekf_fused_ci)
    all_nees_naive.append(nees_naive)
    all_nees_ci.append(nees_ci)

all_filtered_estimates_ekf_fused_naive = np.array(all_filtered_estimates_ekf_fused_naive)
all_filtered_estimates_ekf_fused_ci = np.array(all_filtered_estimates_ekf_fused_ci)
all_filtered_covariances_ekf_fused_naive = np.array(all_filtered_covariances_ekf_fused_naive)
all_filtered_covariances_ekf_fused_ci = np.array(all_filtered_covariances_ekf_fused_ci)
all_nees_naive = np.array(all_nees_naive)
all_nees_ci = np.array(all_nees_ci)

# Mean NEES for each timestep
mean_nees_naive_over_time = all_nees_naive.mean(axis=0)
mean_nees_ci_over_time = all_nees_ci.mean(axis=0)

# Overall Mean NEES
mean_nees_naive = all_nees_naive.mean()
mean_nees_ci = all_nees_ci.mean()

# Determinant of P at every trial/timestep
det_covariances_naive = np.linalg.det(all_filtered_covariances_ekf_fused_naive)
det_covariances_ci = np.linalg.det(all_filtered_covariances_ekf_fused_ci)
det_ratio = det_covariances_ci / det_covariances_naive

# Mean determinant across trials at each timestep
mean_det_naive = det_covariances_naive.mean(axis=0)
mean_det_ci = det_covariances_ci.mean(axis=0)
mean_det_ratio = det_ratio.mean(axis=0)

# Plotting
state_dim = len(x0_true)

times = np.arange(1, n_steps) * dt

print(f"Naive mean NEES: {mean_nees_naive:.3f} (ideal = {state_dim})")
print(f"CI mean NEES: {mean_nees_ci:.3f} (ideal = {state_dim})")

print(f"Mean final naive det(P): {mean_det_naive[-1]:.6e}")
print(f"Mean final CI det(P): {mean_det_ci[-1]:.6e}")

# Plot NEES over time

plt.figure()
plt.plot(times, mean_nees_naive_over_time, label="Naive Fusion")
plt.plot(times, mean_nees_ci_over_time, label="Covariance Intersection")
plt.axhline(
    state_dim,
    linestyle="--",
    label=f"Ideal NEES = {state_dim}"
)

plt.xlabel("Time (s)")
plt.ylabel("Mean NEES")
plt.title("NEES: Naive Fusion vs Covariance Intersection")
plt.legend()
plt.grid()
plt.savefig("figures/demo_track_fusion_1.png")
plt.show()

# Plot determinant of covariance over time

plt.figure()
plt.plot(times, mean_det_ratio, label="CI / Naive")
plt.axhline(
    1.0,
    linestyle="--",
    label="Equal determinant"
)

plt.xlabel("Time (s)")
plt.ylabel("det(P_CI) / det(P_naive)")
plt.title("Covariance Determinant Ratio: CI vs Naive Fusion")
plt.yscale("log")
plt.legend()
plt.grid()
plt.savefig("figures/demo_track_fusion_2.png")
plt.show()