from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "final_comparison.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.ukf import UnscentedKalmanFilter
from tuning_common_fn import mean_nees_for_sequential, mean_nees_for_track_level
from fusion.fuse.naive import fuse_naive
from fusion.fuse.covariance_intersection import covariance_intersection


# Finds optimal sigma_a (for NEES = 4) for sensor fusion (sequential, track level CI) for UKF/EKF
# Naive will use CI sigma_a

# Define sigma_a_grid and tuning_seeds
sigma_a_grid = config['scenarios']['dual_sensor_fusion']['sigma_a_grid']
tuning_seeds = config['tuning_seeds']

# SEQUENTIAL

# Create brentq function for sequential EKF
ekf_sequential_fn = lambda sa: mean_nees_for_sequential(sa, 
                                                        ExtendedKalmanFilter, 
                                                        config, 
                                                        tuning_seeds) - 4


# Identify bracket to use for brentq through grid
print("EKF Sequential Fusion NEES across sigma_a_grid")
for sa in sigma_a_grid:
    print(f"sigma_a: {sa}")
    try:
        print(f"{ekf_sequential_fn(sa) + 4}")
    except np.linalg.LinAlgError:
        print("diverged")
    print("")

# Use minimize_scalar to find optimal sigma_a (50, 100, 500) (4.67, 4.39, 880.52)
# minimize_scalar is being used instead of brentq as while both KF and UKF diverge under 
# sufficiently large sigma_a in sequential fusion (theres two nonlinear relinearization per step)
# EKF's linearized updates accumulate approximation error across both updates, while UKF's sigma point updates do not.
# As a result there was no bracket where nees crossed below 4, so (50, 500) is used instead (100, 4.39 was closest)

min_sa_result_ekf_seq = minimize_scalar(lambda sa: abs(ekf_sequential_fn(sa)), bounds=(50, 500), method='bounded')
sigma_a_ekf_seq = min_sa_result_ekf_seq.x
sigma_a_ekf_seq_nees = ekf_sequential_fn(sigma_a_ekf_seq) + 4

print(f"Optimal sigma_a_ekf_seq: {sigma_a_ekf_seq}") # 168.91741803063607
print(f"Achieved NEES: {sigma_a_ekf_seq_nees}") # 4.197312046716076
print("---")

# Create brentq function for sequential UKF
ukf_sequential_fn = lambda sa: mean_nees_for_sequential(sa, 
                                                        UnscentedKalmanFilter, 
                                                        config, 
                                                        tuning_seeds) - 4

# Identify bracket to use for brentq through grid
print("UKF Sequential Fusion NEES across sigma_a_grid")
for sa in sigma_a_grid:
    print(f"sigma_a: {sa}")
    try:
        print(f"{ukf_sequential_fn(sa) + 4}")
    except np.linalg.LinAlgError:
        print("diverged")
    print("")

# Use brentq to find optimal sigma_a (1, 5) (4.17, 3.21)
sigma_a_ukf_seq = brentq(ukf_sequential_fn, a=1, b=5)

print(f"Optimal sigma_a_ukf_seq: {sigma_a_ukf_seq}") # 1.3053760171426958
print("---")

# COVARIANCE INTERSECTION

# Create brentq function for CI EKF
ekf_CI_fn = lambda sa: mean_nees_for_track_level(sa, 
                                                ExtendedKalmanFilter,
                                                covariance_intersection,
                                                config, 
                                                tuning_seeds) - 4


# Identify bracket to use for brentq through grid
print("EKF CI Fusion NEES across sigma_a_grid")
for sa in sigma_a_grid:
    print(f"sigma_a: {sa}")
    try:
        print(f"{ekf_CI_fn(sa) + 4}")
    except np.linalg.LinAlgError:
        print("diverged")
    print("")


# Use minimize_scalar to find optimal sigma_a (10, 50, 100) (212.47, 4.94, 20.35)
# minimize_scalar is being used instead of brentq as one of the seeds has the trajectory pass near the radar's position (but not close enough for the floor to activate)
# The small range  causes the Jacobian mapping state to range/bearing to become ill conditioned
# This instability propogates to the fused result through CI
# As a result there was no bracket where nees crossed below 4, so (10, 100) is used instead (50, 4.94 was closest)

min_sa_result_ekf_CI = minimize_scalar(lambda sa: abs(ekf_CI_fn(sa)), bounds=(10, 100), method='bounded')
sigma_a_ekf_CI = min_sa_result_ekf_CI.x
sigma_a_ekf_CI_nees = ekf_CI_fn(sigma_a_ekf_CI) + 4

print(f"Optimal sigma_a_ekf_CI: {sigma_a_ekf_CI}") # 40.41685703456618
print(f"Achieved NEES: {sigma_a_ekf_CI_nees}") # 4.0371846848593655
print("---")

# Create brentq function for CI EKF
ukf_CI_fn = lambda sa: mean_nees_for_track_level(sa, 
                                                UnscentedKalmanFilter,
                                                covariance_intersection,
                                                config, 
                                                tuning_seeds) - 4


# Identify bracket to use for brentq through grid
print("UKF CI Fusion NEES across sigma_a_grid")
for sa in sigma_a_grid:
    print(f"sigma_a: {sa}")
    try:
        print(f"{ukf_CI_fn(sa) + 4}")
    except np.linalg.LinAlgError:
        print("diverged")
    print("")

# Use brentq to find optimal sigma_a (5, 10) (4.92, 3.79)
sigma_a_ukf_CI = brentq(ukf_CI_fn, a=5, b=10)

print(f"Optimal sigma_a_ukf_CI: {sigma_a_ukf_CI}") # 8.52044722532766
print("---")
