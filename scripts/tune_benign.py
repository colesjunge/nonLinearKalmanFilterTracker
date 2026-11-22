from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "final_comparison.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
from scipy.optimize import brentq
from fusion.eval.scenarios import build_benign_scenario
from fusion.eval.filters import build_benign_filter
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.ukf import UnscentedKalmanFilter
from tuning_common_fn import mean_nees_for_cell

# Finds optimal sigma_a (for NEES = 4) for benign EKF, UKF

# Define sigma_a_grid and tuning_seeds
sigma_a_grid = config['scenarios']['benign']['sigma_a_grid']
tuning_seeds = config['tuning_seeds']

# Create brentq function for plain CV EKF
ekf_fn = lambda sa: mean_nees_for_cell(sa, 
                                       build_benign_scenario,
                                       lambda s,t,z: build_benign_filter(s,t,z, ExtendedKalmanFilter, config),
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


# Use brentq to find optimal sigma_a (10, 50) (4.27, 2.98)
sigma_a_ekf = brentq(ekf_fn, a=10, b=50)

print(f"Optimal sigma_a_ekf: {sigma_a_ekf}") # 12.525803771515495
print("---")

# Create brentq function for plain CV UKF
ukf_fn = lambda sa: mean_nees_for_cell(sa, 
                                       build_benign_scenario,
                                       lambda s,t,z: build_benign_filter(s,t,z, UnscentedKalmanFilter, config),
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


# Use brentq to find optimal sigma_a (1, 5) (4.57, 3.07)
sigma_a_ukf = brentq(ukf_fn, a=1, b=5)

print(f"Optimal sigma_a_ukf: {sigma_a_ukf}") # 1.5824095654047916
print("---")
