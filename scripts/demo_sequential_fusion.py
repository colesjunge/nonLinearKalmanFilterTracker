import numpy as np
from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.eval.metrics import position_rmse, nees
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.ukf import UnscentedKalmanFilter
from fusion.models.radar import RadarRangeBearing
from fusion.models.eoir import EOIRBearingOnly
from fusion.fuse.sequential import fuse_sequential
from fusion.eval.viz import plot_covariance_ellipse
import matplotlib.pyplot as plt

# This script is to UKF, EKF, and the fused filter for both

rng = np.random.default_rng(42)  # For reproducibility

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

ekf_eoir = ExtendedKalmanFilter(motion_model, eoir_model, x0=x0_filter, P0=P0)
ekf_radar = ExtendedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)
ekf_fused = ExtendedKalmanFilter(motion_model, eoir_model, x0=x0_filter, P0=P0)

ukf_radar = UnscentedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)
ukf_eoir = UnscentedKalmanFilter(motion_model, eoir_model, x0=x0_filter, P0=P0)
ukf_fused = UnscentedKalmanFilter(motion_model, eoir_model, x0=x0_filter, P0=P0)

# Run the simulation
ground_truth = []

filtered_estimates_ekf_eoir = []
filtered_estimates_ekf_radar = []
filtered_estimates_ekf_fused = []

filtered_estimates_ukf_eoir = []
filtered_estimates_ukf_radar = []
filtered_estimates_ukf_fused = []

# Covariances for later plotting
filtered_covariances_ekf_eoir = []
filtered_covariances_ekf_radar = []
filtered_covariances_ekf_fused = []

filtered_covariances_ukf_eoir = []
filtered_covariances_ukf_radar = []
filtered_covariances_ukf_fused = []

for i in range(1, n_steps):
    ground_truth.append(trajectory[i][:2])

    z_radar = radar_model.h(trajectory[i]) + rng.normal(0, [sigma_range, sigma_bearing])
    z_eoir = eoir_model.h(trajectory[i]) + rng.normal(0, [sigma_bearing])

    # Radar predict/update
    ekf_radar.predict(dt)
    ekf_radar.update(z_radar)
    ukf_radar.predict(dt)
    ukf_radar.update(z_radar)

    # EOIR predict/update
    ekf_eoir.predict(dt)
    ekf_eoir.update(z_eoir)
    ukf_eoir.predict(dt)
    ukf_eoir.update(z_eoir)

    # Fused predict/update
    ekf_fused.predict(dt)
    ukf_fused.predict(dt)
    fuse_sequential(ekf_fused, [(radar_model, z_radar), (eoir_model, z_eoir)])
    fuse_sequential(ukf_fused, [(radar_model, z_radar), (eoir_model, z_eoir)])

    # Filtered Estimates
    filtered_estimates_ekf_eoir.append(ekf_eoir.x[:2])
    filtered_estimates_ekf_radar.append(ekf_radar.x[:2])
    filtered_estimates_ukf_eoir.append(ukf_eoir.x[:2])
    filtered_estimates_ukf_radar.append(ukf_radar.x[:2])
    filtered_estimates_ekf_fused.append(ekf_fused.x[:2])
    filtered_estimates_ukf_fused.append(ukf_fused.x[:2])

    # Filtered Covariances
    filtered_covariances_ekf_eoir.append(ekf_eoir.P[:2, :2].copy())
    filtered_covariances_ekf_radar.append(ekf_radar.P[:2, :2].copy())
    filtered_covariances_ekf_fused.append(ekf_fused.P[:2, :2].copy())

    filtered_covariances_ukf_eoir.append(ukf_eoir.P[:2, :2].copy())
    filtered_covariances_ukf_radar.append(ukf_radar.P[:2, :2].copy())
    filtered_covariances_ukf_fused.append(ukf_fused.P[:2, :2].copy())


# Plotting
snapshot = 49 # This is for snapshot 50 due to 0-index (ground truth is 1)

ground_truth_arr = np.array(ground_truth)
filtered_estimates_ekf_eoir_arr = np.array(filtered_estimates_ekf_eoir)
filtered_estimates_ekf_radar_arr = np.array(filtered_estimates_ekf_radar)
filtered_estimates_ukf_eoir_arr = np.array(filtered_estimates_ukf_eoir)
filtered_estimates_ukf_radar_arr = np.array(filtered_estimates_ukf_radar)
filtered_estimates_ekf_fused_arr = np.array(filtered_estimates_ekf_fused)
filtered_estimates_ukf_fused_arr = np.array(filtered_estimates_ukf_fused)

# Calculate RMSE between ground truth and filtered estimates
filtered_ekf_eoir_rmse = position_rmse(filtered_estimates_ekf_eoir_arr , ground_truth_arr)
print(f"Root Mean Square Error (Filtered Estimates EKF EOIR): {filtered_ekf_eoir_rmse}")

filtered_ekf_radar_rmse = position_rmse(filtered_estimates_ekf_radar_arr , ground_truth_arr)
print(f"Root Mean Square Error (Filtered Estimates EKF Radar): {filtered_ekf_radar_rmse}")

filtered_ukf_eoir_rmse = position_rmse(filtered_estimates_ukf_eoir_arr , ground_truth_arr)
print(f"Root Mean Square Error (Filtered Estimates UKF EOIR): {filtered_ukf_eoir_rmse}")

filtered_ukf_radar_rmse = position_rmse(filtered_estimates_ukf_radar_arr , ground_truth_arr)
print(f"Root Mean Square Error (Filtered Estimates UKF Radar): {filtered_ukf_radar_rmse}")

filtered_ekf_fused_rmse = position_rmse(filtered_estimates_ekf_fused_arr , ground_truth_arr)
print(f"Root Mean Square Error (Filtered Estimates EKF Fused): {filtered_ekf_fused_rmse}")

filtered_ukf_fused_rmse = position_rmse(filtered_estimates_ukf_fused_arr , ground_truth_arr)
print(f"Root Mean Square Error (Filtered Estimates UKF Fused): {filtered_ukf_fused_rmse}")


# Covariance determinants at each step
print(f"Covariance determinants at step {snapshot + 1}:")

print(f"EKF EOIR: {np.linalg.det(filtered_covariances_ekf_eoir[snapshot]):.6e}")
print(f"EKF Radar: {np.linalg.det(filtered_covariances_ekf_radar[snapshot]):.6e}")

print(f"UKF EOIR: {np.linalg.det(filtered_covariances_ukf_eoir[snapshot]):.6e}")
print(f"UKF Radar: {np.linalg.det(filtered_covariances_ukf_radar[snapshot]):.6e}")

print(f"UKF Fused: {np.linalg.det(filtered_covariances_ukf_fused[snapshot]):.6e}")
print(f"EKF Fused: {np.linalg.det(filtered_covariances_ekf_fused[snapshot]):.6e}")


fig, ax = plt.subplots(figsize=(8, 8))

# Plot covariance ellipses for EKF

# EKF Radar
plot_covariance_ellipse(
    ax,
    filtered_estimates_ekf_radar[snapshot],
    filtered_covariances_ekf_radar[snapshot],
    n_std=2,
    color="red",
    alpha=0.25,
    label="EKF Radar"
)

# EKF EOIR
plot_covariance_ellipse(
    ax,
    filtered_estimates_ekf_eoir[snapshot],
    filtered_covariances_ekf_eoir[snapshot],
    n_std=2,
    color="blue",
    alpha=0.25,
    label="EKF EOIR"
)

# EKF Fused
plot_covariance_ellipse(
    ax,
    filtered_estimates_ekf_fused[snapshot],
    filtered_covariances_ekf_fused[snapshot],
    n_std=2,
    color="green",
    alpha=0.25,
    label="EKF Fused"
)

# True position at this timestep
ax.scatter(
    trajectory[snapshot + 1][0],
    trajectory[snapshot + 1][1],
    color="black",
    marker="x",
    s=80,
    label="True Position"
)

ax.set_xlabel("x position")
ax.set_ylabel("y position")
ax.set_title("EKF Position Uncertainty at Step 10")
ax.axis("equal")
ax.grid(True)
ax.legend()
plt.savefig("figures/demo_sequential_fusion_1.png")
plt.show()


# Plot covariance ellipses for UKF

fig, ax = plt.subplots(figsize=(8, 8))

# UKF Radar
plot_covariance_ellipse(
    ax,
    filtered_estimates_ukf_radar[snapshot],
    filtered_covariances_ukf_radar[snapshot],
    n_std=2,
    color="red",
    alpha=0.25,
    label="UKF Radar"
)

# UKF EOIR
plot_covariance_ellipse(
    ax,
    filtered_estimates_ukf_eoir[snapshot],
    filtered_covariances_ukf_eoir[snapshot],
    n_std=2,
    color="blue",
    alpha=0.25,
    label="UKF EOIR"
)

# UKF Fused
plot_covariance_ellipse(
    ax,
    filtered_estimates_ukf_fused[snapshot],
    filtered_covariances_ukf_fused[snapshot],
    n_std=2,
    color="green",
    alpha=0.25,
    label="UKF Fused"
)

# True position
ax.scatter(
    trajectory[snapshot + 1][0],
    trajectory[snapshot + 1][1],
    color="black",
    marker="x",
    s=80,
    label="True Position"
)

ax.set_xlabel("x position")
ax.set_ylabel("y position")
ax.set_title("UKF Position Uncertainty at Step 10")
ax.axis("equal")
ax.grid(True)
ax.legend()
plt.savefig("figures/demo_sequential_fusion_2.png")
plt.show()




