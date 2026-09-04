import numpy as np
from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.eval.metrics import position_rmse
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.models.radar import RadarRangeBearing
import matplotlib.pyplot as plt

# This script is to compare the extended kalman filter with and without wrapping.

rng = np.random.default_rng(42)  # For reproducibility

dt = 0.1
n_steps = 100
x0_true = np.array([-500, -25, 0, 5]) # px, py, vx, vy

# Ground truth trajectory
trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

x0_filter = np.array([-450, -90, -1, 7]) # Initial state guess (random numbers I've chosen)
P0 = np.array([[100, 0, 0, 0], 
                [0, 120, 0, 0], 
                [0, 0, 34, 0], 
                [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)

# Filter setup (RadarSensor will be developed later so for now z will be computed directly)
motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
sigma_range = 20.0
sigma_bearing = 0.01
radar_model_with_wrapping = RadarRangeBearing(sigma_range=sigma_range, sigma_bearing=sigma_bearing)
radar_model_without_wrapping = RadarRangeBearing(sigma_range=sigma_range, sigma_bearing=sigma_bearing)
radar_model_without_wrapping.angle_indices = []  # Disable angle wrapping for this model

ekf_with_wrapping = ExtendedKalmanFilter(motion_model, radar_model_with_wrapping, x0=x0_filter, P0=P0)
ekf_without_wrapping = ExtendedKalmanFilter(motion_model, radar_model_without_wrapping, x0=x0_filter, P0=P0)

# Run the simulation
ground_truth = []
filtered_estimates_wrapping = []
filtered_estimates_without_wrapping = []


for i in range(1, n_steps):
    ground_truth.append(trajectory[i][:2])

    z = radar_model_with_wrapping.h(trajectory[i]) + rng.normal(0, [sigma_range, sigma_bearing])

    ekf_with_wrapping.predict(dt)
    ekf_with_wrapping.update(z)
    ekf_without_wrapping.predict(dt)
    ekf_without_wrapping.update(z)

    filtered_estimates_wrapping.append(ekf_with_wrapping.x[:2])
    filtered_estimates_without_wrapping.append(ekf_without_wrapping.x[:2])

ground_truth_arr = np.array(ground_truth)
filtered_estimates_wrapping_arr = np.array(filtered_estimates_wrapping)
filtered_estimates_without_wrapping_arr = np.array(filtered_estimates_without_wrapping)

# Calculate RMSE between ground truth and filtered estimates
filtered_rmse = position_rmse(filtered_estimates_wrapping_arr, ground_truth_arr)
print(f"Root Mean Square Error (Filtered Estimates with Wrapping): {filtered_rmse}")

filtered_rmse_without_wrapping = position_rmse(filtered_estimates_without_wrapping_arr, ground_truth_arr)
print(f"Root Mean Square Error (Filtered Estimates without Wrapping): {filtered_rmse_without_wrapping}")

# Plot error vs time for both EKFs
times = np.arange(1, n_steps) * dt
error_wrapping = np.linalg.norm(filtered_estimates_wrapping_arr - ground_truth_arr, axis=1)
error_without_wrapping = np.linalg.norm(filtered_estimates_without_wrapping_arr - ground_truth_arr, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(
    times,
    error_wrapping,
    label="EKF with wrapping"
)
plt.plot(
    times,
    error_without_wrapping,
    label="EKF without wrapping"
)
plt.xlabel("Time (s)")
plt.ylabel("Position Error")
plt.title("Position Error vs. Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/demo_angle_wrapping_1.png")
plt.show()

# Plot true bearing vs time
true_bearings = np.array([
    radar_model_with_wrapping.h(trajectory[j])[1]
    for j in range(1, n_steps)
])
true_bearings_deg = np.degrees(true_bearings)

plt.figure(figsize=(10, 6))
plt.plot(
    times,
    true_bearings_deg,
    label="True bearing"
)
plt.axhline(180, linestyle=":")
plt.axhline(-180, linestyle=":")
plt.xlabel("Time (s)")
plt.ylabel("Bearing (degrees)")
plt.title("True Bearing vs. Time")
plt.ylim(-200, 200)
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/demo_angle_wrapping_2.png")
plt.show()