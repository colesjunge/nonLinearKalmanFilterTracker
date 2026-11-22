from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "final_comparison.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
from scipy.optimize import minimize_scalar
from fusion.eval.scenarios import build_contaminated_scenario
from fusion.eval.filters import build_contaminated_filter
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.ukf import UnscentedKalmanFilter
from tuning_common_fn import mean_nees_for_cell

# Finds optimal sigma_a (for NEES = 4) for contaminated EKF, UKF

# Define sigma_a_grid and tuning_seeds
sigma_a_grid = config['scenarios']['contaminated']['sigma_a_grid']
tuning_seeds = config['tuning_seeds']

# Create brentq function for plain CV EKF
ekf_fn = lambda sa: mean_nees_for_cell(sa, 
                                       build_contaminated_scenario,
                                       lambda s,t,z: build_contaminated_filter(s,t,z, ExtendedKalmanFilter, config),
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

# Use minimize_scalar to find optimal sigma_a (1, 5, 10) (6068.46, 4420.48, 6758.19)
# minimize_scalar is being used instead of brentq as the contaminated range draws with the tightly trusted bearing
# push the EKF's linearization point to where the Jacobian evaluation never recovers
# As a result there was no bracket where nees crossed below 4, so (1, 10) is used instead (5, 4420.48 was closest)

min_sa_result = minimize_scalar(lambda sa: abs(ekf_fn(sa)), bounds=(1, 10), method='bounded')
sigma_a_ekf = min_sa_result.x
sigma_a_ekf_nees = ekf_fn(sigma_a_ekf) + 4

print(f"Optimal sigma_a_ekf: {sigma_a_ekf}") # 4.920872059510148
print(f"Achieved NEES: {sigma_a_ekf_nees}") # 3775.21182844769

# Create brentq function for plain CV UKF
ukf_fn = lambda sa: mean_nees_for_cell(sa, 
                                       build_contaminated_scenario,
                                       lambda s,t,z: build_contaminated_filter(s,t,z, UnscentedKalmanFilter, config),
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

# UKF cannot complete a single rollout under this contamination level for any tested sigma_a
# EKF degrades badly under the same contamination but never crashes
# The reasoning is that UKF's sigma-point covariance loses positive-definiteness outright
# Thus sigma_a is not applicable

sigma_a_ukf = None
print("Optimal sigma_a_ukf: no sigma_a survived a full rollout under this contamination level")