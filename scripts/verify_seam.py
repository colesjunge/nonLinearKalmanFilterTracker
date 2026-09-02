import numpy as np
from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D
from tracking.filters.kalman import KalmanFilter
from tracking.sim.sensors import PositionSensor
from tracking.eval.metrics import position_rmse
from tracking.eval.viz import plot_tracks

# This script is to verify that tracking imports as expected from the prior tracking project.
# Everything below is precisely run_demo.py from the prior project.

rng = np.random.default_rng(42)  # For reproducibility

dt = 0.1
n_steps = 100
x0_true = np.array([0, 0, 1, 1]) # px, py, vx, vy

# Ground truth trajectory
trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

x0_filter = np.array([.5, .3, 5, 10]) # Initial state uncertainty (random numbers I've chosen)
P0 = np.array([[10, 0, 0, 0], 
                [0, 12, 0, 0], 
                [0, 0, 34, 0], 
                [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)

# Filter setup
motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
measurement_model = LinearPosition2D(sigma_pos=4.0) # Measurement noise
kf = KalmanFilter(motion_model, measurement_model, x0=x0_filter, P0=P0)

# Position sensor setup
position_sensor = PositionSensor(sigma_pos=4.0, p_detect=1.0, clutter_rate=0.0)

# Run the simulation
ground_truth = []
filtered_estimates = []
raw_measurements = []


for i in range(1, n_steps):
    ground_truth.append(trajectory[i][:2])

    measurements = position_sensor.observe(trajectory[i:i+1], t=i*dt, rng=rng)

    kf.predict(dt)
    if measurements:
        kf.update(measurements[0].z)
        raw_measurements.append(measurements[0].z)
    
    filtered_estimates.append(kf.x[:2])

ground_truth_arr = np.array(ground_truth)
filtered_estimates_arr = np.array(filtered_estimates)
raw_measurements_arr = np.array(raw_measurements)

# Calculate RMSE between ground truth and filtered estimates / raw measurements
filtered_rmse = position_rmse(filtered_estimates_arr, ground_truth_arr)
print(f"Root Mean Square Error (Filtered Estimates): {filtered_rmse}")

raw_rmse = position_rmse(raw_measurements_arr, ground_truth_arr)
print(f"Root Mean Square Error (Raw Measurements): {raw_rmse}")

plot_tracks([raw_measurements_arr], [filtered_estimates_arr], [ground_truth_arr], save_path="figures/verify_seam.png")

