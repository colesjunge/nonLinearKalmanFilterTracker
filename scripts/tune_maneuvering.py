from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "final_comparison.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from fusion.eval.scenarios import build_maneuvering_scenario
from fusion.eval.filters import build_maneuvering_filter, build_ct_mode_filter
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.ukf import UnscentedKalmanFilter
from tuning_common_fn import mean_nees_for_cell, compute_omega, straight_mask, ct_plus_mask, ct_minus_mask

# Finds optimal sigma_a (for NEES = 4) for maneuvering EKF, UKF, and IMM Modes (CV, CT plus, CT minus)
# Also finds optimal omega for IMM
# IMM built off of EKF (not UKF)

# Define sigma_a_grid and tuning_seeds
sigma_a_grid = config['scenarios']['maneuvering']['sigma_a_grid']
tuning_seeds = config['tuning_seeds']

# Create brentq function for plain CV EKF
ekf_fn = lambda sa: mean_nees_for_cell(sa, 
                                       build_maneuvering_scenario,
                                       lambda s,t,z: build_maneuvering_filter(s,t,z, ExtendedKalmanFilter, config),
                                       config, 
                                       tuning_seeds) - 4

# Identify bracket to use for brentq through grid
print("EKF NEES across sigma_a_grid")
for sa in sigma_a_grid:
    print(f"sigma_a: {sa}")
    try:
        print(f"{ekf_fn(sa) + 4}")
    except np.linalg.LinAlgError:
        print("diverged")
    print("")

# Use brentq to find optimal sigma_a (10, 50) (1095.2, 3.41)
sigma_a_ekf = brentq(ekf_fn, a=10, b=50)

print(f"Optimal sigma_a_ekf: {sigma_a_ekf}") # 41.46085564065797
print("---")

# Create brentq function for plain CV UKF
ukf_fn = lambda sa: mean_nees_for_cell(sa, 
                                       build_maneuvering_scenario,
                                       lambda s,t,z: build_maneuvering_filter(s,t,z, UnscentedKalmanFilter, config),
                                       config, 
                                       tuning_seeds) - 4

# Identify bracket to use for brentq through grid
print("UKF NEES across sigma_a_grid")
for sa in sigma_a_grid:
    print(f"sigma_a: {sa}")
    try:
        print(f"{ukf_fn(sa) + 4}")
    except np.linalg.LinAlgError:
        print("diverged")
    print("")

# Use brentq to find optimal sigma_a (10, 50) (42.05, 3.49)
sigma_a_ukf = brentq(ukf_fn, a=10, b=50)

print(f"Optimal sigma_a_ukf: {sigma_a_ukf}") # 42.715311386770004
print("---")

# Find optimal omega
omega_val = compute_omega(config, tuning_seeds) 

print("")
print(f"Optimal omega: {omega_val}") # 2.8389837680632013
print("---")

# Create brentq function for CV mode for IMM
cv_fn = lambda sa: mean_nees_for_cell(sa, 
                           build_maneuvering_scenario,
                           lambda s,t,z: build_maneuvering_filter(s,t,z, ExtendedKalmanFilter, config),
                           config, 
                           tuning_seeds, 
                           mask_fn=straight_mask) - 4

# Identify bracket to use for brentq through grid
print("CV NEES across sigma_a_grid")
for sa in sigma_a_grid:
    print(f"sigma_a: {sa}")
    try:
        print(f"{cv_fn(sa) + 4}")
    except np.linalg.LinAlgError:
        print("diverged")
    print("")

# Use brentq to find optimal sigma_a (10, 50) (1350.36, 2.90)
sigma_a_cv = brentq(cv_fn, a=10, b=50)

print(f"Optimal sigma_a_cv: {sigma_a_cv}") # 31.89720488088903
print("---")

# Create brentq function for CT CW mode for IMM
ct_plus_fn = lambda sa: mean_nees_for_cell(sa, 
                           build_maneuvering_scenario,
                           lambda s,t,z: build_ct_mode_filter(s,t,z, ExtendedKalmanFilter, omega=omega_val, config=config),
                           config, 
                           tuning_seeds, 
                           mask_fn=ct_plus_mask) - 4

# Identify bracket to use for brentq through grid
print("CT plus NEES across sigma_a_grid")
for sa in sigma_a_grid:
    print(f"sigma_a: {sa}")
    try:
        print(f"{ct_plus_fn(sa) + 4}")
    except np.linalg.LinAlgError:
        print("diverged")
    print("")

# Use brentq to find optimal sigma_a (50, 100) (7.55, 3.37)
sigma_a_ct_plus = brentq(ct_plus_fn, a=50, b=100)

print(f"Optimal sigma_a_ct_plus: {sigma_a_ct_plus}") # 79.05173355984762
print("---")

# Create brentq function for CV CCW mode for IMM
ct_minus_fn = lambda sa: mean_nees_for_cell(sa, 
                           build_maneuvering_scenario,
                           lambda s,t,z: build_ct_mode_filter(s,t,z, ExtendedKalmanFilter, omega=-omega_val, config=config),
                           config, 
                           tuning_seeds, 
                           mask_fn=ct_minus_mask) - 4

# Identify bracket to use for brentq through grid
print("CT minus NEES across sigma_a_grid")
for sa in sigma_a_grid:
    print(f"sigma_a: {sa}")
    try:
        print(f"{ct_minus_fn(sa) + 4}")
    except np.linalg.LinAlgError:
        print("diverged")
    print("")

# Narrowing down for (100, 200, 300, 400, 500)
for sa in [100, 200, 300, 400, 500]:
    print(f"sigma_a: {sa}")
    try:
        print(f"{ct_minus_fn(sa) + 4}")
    except np.linalg.LinAlgError:
        print("diverged")
    print("")

# Use minimize_scalar to find optimal sigma_a (300, 400, 500) (114.7, 8.39, 105.8)
# minimize_scalar is being used instead of brentq as the fixed direction CT filter run on the mixed trajectory
# is systematically biased during straight and opposite turn legs which the sigma_a cannot absorb smoothly
# As a result there was no bracket where nees crossed below 4, so (300, 500) is used instead (400, 8.39 was closest)

min_sa_result = minimize_scalar(lambda sa: abs(ct_minus_fn(sa)), bounds=(300, 500), method='bounded')
sigma_a_ct_minus = min_sa_result.x
sigma_a_ct_minus_nees = ct_minus_fn(sigma_a_ct_minus) + 4

print(f"Optimal sigma_a_ct_minus: {sigma_a_ct_minus}") # 401.82066209219937
print(f"Achieved NEES: {sigma_a_ct_minus_nees}") # 8.095285086913595
