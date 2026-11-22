from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "final_comparison.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
from scipy.optimize import minimize_scalar
from fusion.eval.scenarios import build_bearing_only_scenario
from fusion.eval.filters import build_bearing_only_filter
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.ukf import UnscentedKalmanFilter
from tuning_common_fn import mean_nees_for_cell

# Finds optimal sigma_a (for NEES = 4) for contaminated bearing only EKF, UKF

# Define sigma_a_grid and tuning_seeds
sigma_a_grid = config['scenarios']['bearing_only']['sigma_a_grid']
tuning_seeds = config['tuning_seeds']

# Create brentq function for plain CV EKF
ekf_fn = lambda sa: mean_nees_for_cell(sa, 
                                       build_bearing_only_scenario,
                                       lambda s,t,z: build_bearing_only_filter(s,t,z, ExtendedKalmanFilter, config),
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


# Use minimize_scalar to find optimal sigma_a (10, 50, 100, 500 1000) (196.74, 172.04, 179.67, 171.97, 175.06)
# minimize_scalar is being used instead of brentq as bearing-only geometry can't directly resolve range (only through slow triangulation)
# The contaminated noise compounds this difficulty even more 
# As a result there was no bracket where nees crossed below 4, so (10, 1000) is used instead (500, 171.97 was closest)

min_sa_result_ekf = minimize_scalar(lambda sa: abs(ekf_fn(sa)), bounds=(10, 1000), method='bounded')
sigma_a_ekf = min_sa_result_ekf.x
sigma_a_ekf_nees = ekf_fn(sigma_a_ekf) + 4

print(f"Optimal sigma_a_ekf: {sigma_a_ekf}") # 634.404541978951
print(f"Achieved NEES: {sigma_a_ekf_nees}") # 168.61640085908238


# Create brentq function for plain CV UKF
ukf_fn = lambda sa: mean_nees_for_cell(sa, 
                                       build_bearing_only_scenario,
                                       lambda s,t,z: build_bearing_only_filter(s,t,z, UnscentedKalmanFilter, config),
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

# Use minimize_scalar to find optimal sigma_a (.1, .5, 1, 5) (81.7, 78.68, 78.83, 82.98)
# minimize_scalar is being used instead of brentq as bearing-only geometry can't directly resolve range (only through slow triangulation)
# The contaminated noise compounds this difficulty even more 
# As a result there was no bracket where nees crossed below 4, so (.1, 5) is used instead (.5, 78.68 was closest)

min_sa_result_ukf = minimize_scalar(lambda sa: abs(ukf_fn(sa)), bounds=(.1, 5), method='bounded')
sigma_a_ukf = min_sa_result_ukf.x
sigma_a_ukf_nees = ukf_fn(sigma_a_ukf) + 4

print(f"Optimal sigma_a_ukf: {sigma_a_ukf}") # 0.6631614438565597
print(f"Achieved NEES: {sigma_a_ukf_nees}") # 78.57123492746203