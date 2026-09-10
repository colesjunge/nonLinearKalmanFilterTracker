import numpy as np
from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.eval.metrics import position_rmse, nees
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.ukf import UnscentedKalmanFilter
from fusion.models.radar import RadarRangeBearing
from fusion.sim.noise import contaminated_gaussian_noise
import matplotlib.pyplot as plt

# This script is to compare the degradation of the extended kalman filter and the unscented kalman filter under contaminated noise
# No fusion or EOIR will be incorporated as it is not the focus

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

# Filter setup
motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
sigma_range = 100.0 # Higher to test bearing only (range measurements effectively noise)
sigma_bearing = 0.01
radar_model= RadarRangeBearing(sigma_range=sigma_range, sigma_bearing=sigma_bearing)


# To collect data from multiple trials
def run_trial(seed) -> tuple[list[np.ndarray], 
                             list[np.ndarray], 
                             list[np.ndarray], 
                             list[np.ndarray], 
                             list[float], 
                             list[float], 
                             list[float], 
                             list[float], 
                             list[np.ndarray]]:
    
    rng = np.random.default_rng(seed)  # For reproducibility
    ekf_clean= ExtendedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)
    ukf_clean= UnscentedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)
    ekf_contaminated = ExtendedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)
    ukf_contaminated = UnscentedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)

    ground_truth = []
    filtered_estimates_ekf_clean = []
    filtered_estimates_ukf_clean = []
    filtered_estimates_ekf_contaminated = []
    filtered_estimates_ukf_contaminated = []
    nees_ekf_clean = []
    nees_ukf_clean = []
    nees_ekf_contaminated = []
    nees_ukf_contaminated = []

    for i in range(1, n_steps):
        ground_truth.append(trajectory[i][:2])

        z_clean = radar_model.h(trajectory[i]) + rng.normal(0, [sigma_range, sigma_bearing])
        z_contaminated = radar_model.h(trajectory[i]) + contaminated_gaussian_noise(rng, radar_model.R()) # Epsilon and Kappa kept at defaults (.1 and 10) for this

        ekf_clean.predict(dt)
        ekf_clean.update(z_clean)
        ukf_clean.predict(dt)
        ukf_clean.update(z_clean)

        ekf_contaminated.predict(dt)
        ekf_contaminated.update(z_contaminated)
        ukf_contaminated.predict(dt)
        ukf_contaminated.update(z_contaminated)

        filtered_estimates_ekf_clean.append(ekf_clean.x[:2])
        filtered_estimates_ukf_clean.append(ukf_clean.x[:2])
        filtered_estimates_ekf_contaminated.append(ekf_contaminated.x[:2])
        filtered_estimates_ukf_contaminated.append(ukf_contaminated.x[:2])

        nees_ekf_clean.append(nees(ekf_clean.x, ekf_clean.P, trajectory[i]))
        nees_ukf_clean.append(nees(ukf_clean.x, ukf_clean.P, trajectory[i]))
        nees_ekf_contaminated.append(nees(ekf_contaminated.x, ekf_contaminated.P, trajectory[i]))
        nees_ukf_contaminated.append(nees(ukf_contaminated.x, ukf_contaminated.P, trajectory[i]))

    return filtered_estimates_ekf_clean, filtered_estimates_ukf_clean, filtered_estimates_ekf_contaminated, filtered_estimates_ukf_contaminated, nees_ekf_clean, nees_ukf_clean, nees_ekf_contaminated, nees_ukf_contaminated, ground_truth

all_filtered_estimates_ekf_clean = []
all_filtered_estimates_ukf_clean = []
all_filtered_estimates_ekf_contaminated= []
all_filtered_estimates_ukf_contaminated = []
all_nees_ekf_clean = []
all_nees_ukf_clean = []
all_nees_ekf_contaminated= []
all_nees_ukf_contaminated = []
all_ground_truth = []

# Run the simulation
for trial in range(n_trials): 
    filtered_estimates_ekf_clean, filtered_estimates_ukf_clean, filtered_estimates_ekf_contaminated, filtered_estimates_ukf_contaminated, nees_ekf_clean, nees_ukf_clean, nees_ekf_contaminated, nees_ukf_contaminated, ground_truth = run_trial(trial)

    all_filtered_estimates_ekf_clean.append(filtered_estimates_ekf_clean)
    all_filtered_estimates_ukf_clean.append(filtered_estimates_ukf_clean)
    all_filtered_estimates_ekf_contaminated.append(filtered_estimates_ekf_contaminated)
    all_filtered_estimates_ukf_contaminated.append(filtered_estimates_ukf_contaminated)
    all_nees_ekf_clean.append(nees_ekf_clean)
    all_nees_ukf_clean.append(nees_ukf_clean)
    all_nees_ekf_contaminated.append(nees_ekf_contaminated)
    all_nees_ukf_contaminated.append(nees_ukf_contaminated)
    all_ground_truth.append(ground_truth)


ground_truth_arr_3d = np.array(all_ground_truth)
ground_truth_arr_2d = ground_truth_arr_3d.reshape(-1, 2)  # Reshape to (n_trials * n_steps, 2)

filtered_estimates_ekf_clean_arr_3d = np.array(all_filtered_estimates_ekf_clean)
filtered_estimates_ekf_clean_arr_2d = filtered_estimates_ekf_clean_arr_3d.reshape(-1, 2)  # Reshape to (n_trials * n_steps, 2)

filtered_estimates_ukf_clean_arr_3d = np.array(all_filtered_estimates_ukf_clean)
filtered_estimates_ukf_clean_arr_2d = filtered_estimates_ukf_clean_arr_3d.reshape(-1, 2)  # Reshape to (n_trials * n_steps, 2)

filtered_estimates_ekf_contaminated_arr_3d = np.array(all_filtered_estimates_ekf_contaminated)
filtered_estimates_ekf_contaminated_arr_2d = filtered_estimates_ekf_contaminated_arr_3d.reshape(-1, 2)  # Reshape to (n_trials * n_steps, 2)

filtered_estimates_ukf_contaminated_arr_3d = np.array(all_filtered_estimates_ukf_contaminated)
filtered_estimates_ukf_contaminated_arr_2d = filtered_estimates_ukf_contaminated_arr_3d.reshape(-1, 2)  # Reshape to (n_trials * n_steps, 2)

mean_ekf_clean = np.mean(all_nees_ekf_clean)
mean_ukf_clean = np.mean(all_nees_ukf_clean)
mean_ekf_contaminated = np.mean(all_nees_ekf_contaminated)
mean_ukf_contaminated= np.mean(all_nees_ukf_contaminated)

# Calculate RMSE between ground truth and filtered estimates
filtered_rmse_ekf_clean = position_rmse(filtered_estimates_ekf_clean_arr_2d, ground_truth_arr_2d)
print(f"Root Mean Square Error (EKF Filtered Estimates with no Contamination): {filtered_rmse_ekf_clean}")

filtered_rmse_ukf_clean = position_rmse(filtered_estimates_ukf_clean_arr_2d, ground_truth_arr_2d)
print(f"Root Mean Square Error (UKF Filtered Estimates with no Contamination): {filtered_rmse_ukf_clean}")

filtered_rmse_ekf_contaminated = position_rmse(filtered_estimates_ekf_contaminated_arr_2d, ground_truth_arr_2d)
print(f"Root Mean Square Error (EKF Filtered Estimates with Contamination): {filtered_rmse_ekf_contaminated}")

filtered_rmse_ukf_contaminated = position_rmse(filtered_estimates_ukf_contaminated_arr_2d, ground_truth_arr_2d)
print(f"Root Mean Square Error (UKF Filtered Estimates with Contamination): {filtered_rmse_ukf_contaminated}")


# Calculate Average NEES for both filters
print(f"Average NEES (EKF; No Contamination): {mean_ekf_clean}")
print(f"Average NEES (UKF; No Contamination): {mean_ukf_clean}")

print(f"Average NEES (EKF; Contamination): {mean_ekf_contaminated}")
print(f"Average NEES (UKF; Contamination): {mean_ukf_contaminated}")

# Plot NEES vs times
times = np.arange(1, n_steps) * dt

plt.figure(figsize=(10, 6))
plt.plot(
    times,
    np.mean(np.array(all_nees_ekf_clean), axis=0),
    label="EKF NEES (No Contamination)"
)
plt.plot(
    times,
    np.mean(np.array(all_nees_ukf_clean), axis=0),
    label="UKF NEES (No Contamination)"
)
plt.plot(
    times,
    np.mean(np.array(all_nees_ekf_contaminated), axis=0),
    label="EKF NEES (Contamination)"
)
plt.plot(
    times,
    np.mean(np.array(all_nees_ukf_contaminated), axis=0),
    label="UKF NEES (Contamination)"
)
plt.axhline(y=4, color='r', linestyle=':', label="NEES Threshold (dim_x)")
plt.xlabel("Time (s)")
plt.ylabel("NEES")
plt.title("NEES vs. Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/demo_contaminated_noise_1.png")
plt.show()

# Plot error vs time for both filters
times = np.arange(1, n_steps) * dt
error_ekf_clean = np.mean(np.linalg.norm(filtered_estimates_ekf_clean_arr_3d - ground_truth_arr_3d, axis=2), axis=0)
error_ukf_clean = np.mean(np.linalg.norm(filtered_estimates_ukf_clean_arr_3d - ground_truth_arr_3d, axis=2), axis=0)
error_ekf_contaminated = np.mean(np.linalg.norm(filtered_estimates_ekf_contaminated_arr_3d - ground_truth_arr_3d, axis=2), axis=0)
error_ukf_contaminated = np.mean(np.linalg.norm(filtered_estimates_ukf_contaminated_arr_3d - ground_truth_arr_3d, axis=2), axis=0)

plt.figure(figsize=(10, 6))
plt.plot(
    times,
    error_ekf_clean,
    label="EKF (No Contamination)"
)
plt.plot(
    times,
    error_ukf_clean,
    label="UKF (No Contamination)"
)
plt.plot(
    times,
    error_ekf_contaminated,
    label="EKF (Contamination)"
)
plt.plot(
    times,
    error_ukf_contaminated,
    label="UKF (Contamination)"
)
plt.xlabel("Time (s)")
plt.ylabel("Position Error")
plt.title("Position Error vs. Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/demo_contaminated_noise_2.png")
plt.show()