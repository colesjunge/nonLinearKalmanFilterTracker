from pathlib import Path
import yaml

config_imm_path = Path(__file__).resolve().parent.parent / "configs" / "imm.yaml"
with open(config_imm_path) as f:
    config_imm = yaml.safe_load(f)

config_training_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_training_path) as f:
    config_training = yaml.safe_load(f)

import numpy as np
from scipy.optimize import brentq
from typing import Callable

from tracking.models.motion import ConstantVelocity2D, MotionModel
from tracking.models.measurement import LinearPosition2D
from tracking.filters.kalman import KalmanFilter
from tracking.filters.utils import wrap_angle

from fusion.models.motion import CoordinatedTurn2D
from fusion.eval.metrics import classify_maneuver_segments
from fusion.eval.rollout import run_filter_on_trajectory
from fusion.learned.datasets import load_maneuver_dataset

# This script is to tune sigma_a for CV baseline, IMM (only CV), and IMM (both CTs)
# Also will find optimal omega

dt = config_imm['dt']
sigma_pos = config_imm['sigma_pos']
sigma_a_grid = config_imm['tuning']['sigma_a_grid']
tuning_seeds = config_imm['tuning']['tuning_seeds']
P0 = np.array(config_imm['initialization']['P0'])

# Load train/validation data from prior created dataset (K is not relevant as windows not neccessary for this script, only val_dataset.targets)
_, val_dataset = load_maneuver_dataset(
           path='data/processed/maneuver_dataset.npz', 
           K=1,
           val_fraction=config_training['maneuver']['val_fraction'], 
           seed=config_training['seed'],
           sigma_pos=config_training['maneuver']['sigma_pos'])

trajectories = val_dataset.targets[:100]  # (n_val, T, 4), subsample to increase tuning speed

# Calculate mean NEES for each sigma_a
def mean_nees_for_sigma_a(sigma_a: float,
                        motion_model_fn: Callable[[float], MotionModel],
                        mask_fn: Callable[[np.ndarray], np.ndarray] | None,
                        trajectories: np.ndarray, 
                        tuning_seeds: list[int],
                        dt: float,
                        sigma_pos: float,
                        P0: np.ndarray) -> float:

    T = trajectories.shape[1]

    # Accumulate NEES across all tuning seeds
    sigma_a_nees = []

    for seed in tuning_seeds:
        rng = np.random.default_rng(seed)

        for trajectory in trajectories:
            z_seq = trajectory[:, :2] + rng.normal(0, sigma_pos, size=(T, 2))

            x0 = np.concatenate([z_seq[0], np.zeros(2)])

            # Create filter and run on trajectory to get nees
            filter = KalmanFilter(motion_model_fn(sigma_a), LinearPosition2D(sigma_pos), x0, P0)

            _, nees_seq = run_filter_on_trajectory(filter, trajectory, z_seq, dt)

            # If mask present (for turn vs straight legs), apply
            if mask_fn is not None:
                mask = mask_fn(trajectory)
                nees_seq = nees_seq[mask]

            sigma_a_nees.extend(nees_seq)

    # Compute and return mean nees
    mean_nees = np.mean(sigma_a_nees)

    return mean_nees

# Find optimal omega
all_turning_angular_rates = []

for trajectory in trajectories:

    mask, angular_rates = classify_maneuver_segments(trajectory, dt)

    turning_angular_rates = angular_rates[mask]

    all_turning_angular_rates.append(turning_angular_rates)

all_turning_angular_rates = np.abs(np.concatenate(all_turning_angular_rates))

# Set optimal omega to median of abs value of rates
omega = np.median(all_turning_angular_rates)
print(f"omega={omega}")

# Run sweep over sigma_a values
all_pairs_base = []
all_pairs_cv_imm = []
all_pairs_ct_imm = []

# Motion models functions
cv_fn = lambda sa: ConstantVelocity2D(sa)
ct_fn = lambda sa: CoordinatedTurn2D(omega=omega, sigma_a=sa)

# Mask functions
cv_imm_mask_fn = lambda traj: ~classify_maneuver_segments(traj, dt)[0]
ct_imm_mask_fn = lambda traj: classify_maneuver_segments(traj, dt)[0]

for sigma_a in sigma_a_grid:

    # Get current mean NEES for cv baseline
    curr_mean_nees_cv_baseline =  mean_nees_for_sigma_a(sigma_a=sigma_a,
                                            motion_model_fn=cv_fn,
                                            mask_fn=None,
                                            trajectories=trajectories,
                                            tuning_seeds=tuning_seeds,
                                            dt=dt,
                                            sigma_pos=sigma_pos,
                                            P0=P0)

    # Get current mean NEES for cv baseline
    curr_mean_nees_cv_imm = mean_nees_for_sigma_a(sigma_a=sigma_a,
                                            motion_model_fn=cv_fn,
                                            mask_fn=cv_imm_mask_fn,
                                            trajectories=trajectories,
                                            tuning_seeds=tuning_seeds,
                                            dt=dt,
                                            sigma_pos=sigma_pos,
                                            P0=P0)
    # Get current mean NEES for cv baseline
    curr_mean_nees_ct_imm =  mean_nees_for_sigma_a(sigma_a=sigma_a,
                                            motion_model_fn=ct_fn,
                                            mask_fn=ct_imm_mask_fn,
                                            trajectories=trajectories,
                                            tuning_seeds=tuning_seeds,
                                            dt=dt,
                                            sigma_pos=sigma_pos,
                                            P0=P0)

    # Accumulate pairs (sigma_a, mean_nees) and print ongoing results
    print(f"sigma_a={sigma_a}, cv_baseline={curr_mean_nees_cv_baseline:.4f}")
    all_pairs_base.append((sigma_a, curr_mean_nees_cv_baseline))

    print(f"sigma_a={sigma_a}, cv_imm={curr_mean_nees_cv_imm:.4f}")
    all_pairs_cv_imm.append((sigma_a, curr_mean_nees_cv_imm))

    print(f"sigma_a={sigma_a}, ct_imm={curr_mean_nees_ct_imm:.4f}")
    all_pairs_ct_imm.append((sigma_a, curr_mean_nees_ct_imm))

# From sigma_a sweep, brackets will be as follows (Optimal NEES is 4)
# Baseline [10, 50] (28.9, 3.31)
# CV IMM [10, 50] (20.4, 2.64)
# CT IMM [50, 100] (7.94, 3.39)

brentq_baseline_fn = lambda sa: mean_nees_for_sigma_a(sa,
                                                    motion_model_fn=cv_fn,
                                                    mask_fn=None,
                                                    trajectories=trajectories,
                                                    tuning_seeds=tuning_seeds,
                                                    dt=dt,
                                                    sigma_pos=sigma_pos,
                                                    P0=P0) - 4

brentq_cv_imm_fn = lambda sa: mean_nees_for_sigma_a(sigma_a=sa,
                                            motion_model_fn=cv_fn,
                                            mask_fn=cv_imm_mask_fn,
                                            trajectories=trajectories,
                                            tuning_seeds=tuning_seeds,
                                            dt=dt,
                                            sigma_pos=sigma_pos,
                                            P0=P0) - 4

brentq_ct_imm_fn = lambda sa: mean_nees_for_sigma_a(sigma_a=sa,
                                            motion_model_fn=ct_fn,
                                            mask_fn=ct_imm_mask_fn,
                                            trajectories=trajectories,
                                            tuning_seeds=tuning_seeds,
                                            dt=dt,
                                            sigma_pos=sigma_pos,
                                            P0=P0) - 4


sigma_a_baseline = brentq(brentq_baseline_fn, a=10, b=50)
sigma_a_cv_imm = brentq(brentq_cv_imm_fn, a=10, b=50)
sigma_a_ct_imm = brentq(brentq_ct_imm_fn, a=50, b=100)

print(f"sigma_a_baseline={sigma_a_baseline}")
print(f"sigma_a_cv_imm={sigma_a_cv_imm}")
print(f"sigma_a_ct_imm={sigma_a_ct_imm}")

# Results have been added to imm.yaml
# omega = 2.57202
# sigma_a_baseline = 39.807465882640216
# sigma_a_cv_imm = 28.758722290341623
# sigma_a_ct_imm = 83.43547142902904