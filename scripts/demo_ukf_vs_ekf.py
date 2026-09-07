import numpy as np
from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.eval.metrics import position_rmse, nees
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.ukf import UnscentedKalmanFilter
from fusion.models.radar import RadarRangeBearing
import matplotlib.pyplot as plt

# This script is to compare the extended kalman filter to the unscented kalman filter
# The intiial states are chosen to reproduce well known bearing, but not range
# Here, the UKF should outperform the EKF because the EKF linearizes the measurement model, which is not accurate for large initial errors in range.

dt = 0.1
n_steps = 100
n_trials = 30
x0_true = np.array([-50.0, -25.0, 0.0, 5.0]) # px, py, vx, vy

# Ground truth trajectory
trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

x0_filter = np.array([-45.0, -26.0, -1.0, 4.5]) # Initial state guess (random numbers I've chosen)
P0 = np.array([[900, 0, 0, 0], 
                [0, 4, 0, 0], 
                [0, 0, 4, 0], 
                [0, 0, 0, 4]]) # Initial uncertainty (random numbers I've chosen)

# Filter setup (RadarSensor will be developed later so for now z will be computed directly)
motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
sigma_range = 100.0 # Higher to test bearing only (range measurements effectively noise)
sigma_bearing = 0.01
radar_model= RadarRangeBearing(sigma_range=sigma_range, sigma_bearing=sigma_bearing)


# To collect data from multiple trials
def run_trial(seed) -> tuple[list[np.ndarray], list[np.ndarray], list[float], list[float], list[np.ndarray]]:
    rng = np.random.default_rng(seed)  # For reproducibility
    ekf= ExtendedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)
    ukf= UnscentedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)

    ground_truth = []
    filtered_estimates_ekf = []
    filtered_estimates_ukf = []
    nees_ekf = []
    nees_ukf = []

    for i in range(1, n_steps):
        ground_truth.append(trajectory[i][:2])

        z = radar_model.h(trajectory[i]) + rng.normal(0, [sigma_range, sigma_bearing])

        ekf.predict(dt)
        ekf.update(z)
        ukf.predict(dt)
        ukf.update(z)

        filtered_estimates_ekf.append(ekf.x[:2])
        filtered_estimates_ukf.append(ukf.x[:2])

        nees_ekf.append(nees(ekf.x, ekf.P, trajectory[i]))
        nees_ukf.append(nees(ukf.x, ukf.P, trajectory[i]))

    return filtered_estimates_ekf, filtered_estimates_ukf, nees_ekf, nees_ukf, ground_truth

all_filtered_estimates_ekf = []
all_filtered_estimates_ukf = []
all_nees_ekf = []
all_nees_ukf = []
all_ground_truth = []

# Run the simulation
for trial in range(n_trials): 
    filtered_estimates_ekf, filtered_estimates_ukf, nees_ekf, nees_ukf, ground_truth = run_trial(trial)

    all_filtered_estimates_ekf.append(filtered_estimates_ekf)
    all_filtered_estimates_ukf.append(filtered_estimates_ukf)
    all_nees_ekf.append(nees_ekf)
    all_nees_ukf.append(nees_ukf)
    all_ground_truth.append(ground_truth)


ground_truth_arr_3d = np.array(all_ground_truth)
ground_truth_arr_2d = ground_truth_arr_3d.reshape(-1, 2)  # Reshape to (n_trials * n_steps, 2)
filtered_estimates_ekf_arr_3d = np.array(all_filtered_estimates_ekf)
filtered_estimates_ekf_arr_2d = filtered_estimates_ekf_arr_3d.reshape(-1, 2)  # Reshape to (n_trials * n_steps, 2)
filtered_estimates_ukf_arr_3d = np.array(all_filtered_estimates_ukf)
filtered_estimates_ukf_arr_2d = filtered_estimates_ukf_arr_3d.reshape(-1, 2)  # Reshape to (n_trials * n_steps, 2)

mean_ekf = np.mean(all_nees_ekf)
mean_ukf = np.mean(all_nees_ukf)

# Calculate RMSE between ground truth and filtered estimates
filtered_rmse_ekf = position_rmse(filtered_estimates_ekf_arr_2d, ground_truth_arr_2d)
print(f"Root Mean Square Error (EKF Filtered Estimates with Wrapping): {filtered_rmse_ekf}")

filtered_rmse_ukf = position_rmse(filtered_estimates_ukf_arr_2d, ground_truth_arr_2d)
print(f"Root Mean Square Error (UKF Filtered Estimates): {filtered_rmse_ukf}")


# Calculate Average NEES for both filters
print(f"Average NEES (EKF): {mean_ekf}")
print(f"Average NEES (UKF): {mean_ukf}")

# Plot NEES vs times
times = np.arange(1, n_steps) * dt

plt.figure(figsize=(10, 6))
plt.plot(
    times,
    np.mean(np.array(all_nees_ekf), axis=0),
    label="EKF NEES"
)
plt.plot(
    times,
    np.mean(np.array(all_nees_ukf), axis=0),
    label="UKF NEES"
)
plt.axhline(y=4, color='r', linestyle=':', label="NEES Threshold (dim_x)")
plt.xlabel("Time (s)")
plt.ylabel("NEES")
plt.title("NEES vs. Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/demo_ukf_vs_ekf_1.png")
plt.show()

# Plot error vs time for both filters
times = np.arange(1, n_steps) * dt
error_ekf = np.mean(np.linalg.norm(filtered_estimates_ekf_arr_3d - ground_truth_arr_3d, axis=2), axis=0)
error_ukf = np.mean(np.linalg.norm(filtered_estimates_ukf_arr_3d - ground_truth_arr_3d, axis=2), axis=0)

plt.figure(figsize=(10, 6))
plt.plot(
    times,
    error_ekf,
    label="EKF"
)
plt.plot(
    times,
    error_ukf,
    label="UKF"
)
plt.xlabel("Time (s)")
plt.ylabel("Position Error")
plt.title("Position Error vs. Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/demo_ukf_vs_ekf_2.png")
plt.show()