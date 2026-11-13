from pathlib import Path
import yaml

config_imm_path = Path(__file__).resolve().parent.parent / "configs" / "imm.yaml"
with open(config_imm_path) as f:
    config_imm = yaml.safe_load(f)

config_training_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_training_path) as f:
    config_training = yaml.safe_load(f)

import numpy as np
import torch
from scipy import linalg
import matplotlib.pyplot as plt

from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D
from tracking.filters.kalman import KalmanFilter

from fusion.models.motion import CoordinatedTurn2D
from fusion.filters.imm import InteractingMultipleModel
from fusion.eval.metrics import classify_maneuver_segments
from fusion.eval.rollout import run_filter_on_trajectory
from fusion.learned.datasets import load_maneuver_dataset
from fusion.learned.checkpoints import load_checkpoint
from fusion.learned.residual_dynamics_kf import build_residual_dynamics_kf  # once it exists


# This script will compare a baseline CV KF, an IMM (CV, CT+, CT-), and a Residual Dynamics KF
# Squared error and NEES are collected for each of the three including for the overall trajectory and straight vs turning legs

dt = config_imm["dt"]
sigma_pos = config_imm["sigma_pos"]
P0 = np.array(config_imm["initialization"]["P0"])
mc_seeds = config_imm["comparison"]["mc_seeds"]

methods = ["CV", "IMM", "ResidualDynamicsKF"]
regimes = ["straight", "turning", "overall"]
metrics = ["sq_err", "nees"]

# Load in val_dataset
_, val_dataset = load_maneuver_dataset(path='data/processed/maneuver_dataset.npz',
                                        K=1,
                                        val_fraction=config_training['maneuver']['val_fraction'],
                                        seed=config_training['seed'],
                                        sigma_pos=config_training['maneuver']['sigma_pos'])
                                       
trajectories = val_dataset.targets # (n_val, T, 4)

# Load in trained checkpoint
model, _, checkpoint_config = load_checkpoint(path='data/processed/residual_dynamics_checkpoint.pt',
                                                build_model_fn=build_residual_dynamics_kf)
model.eval()

# CV filter and IMM are built for each trajectory

def build_cv_filter(sigma_a: float, x0: np.ndarray, P0: np.ndarray) -> KalmanFilter:
    """
    Creates a Kalman filter
    """
    return KalmanFilter(ConstantVelocity2D(sigma_a), LinearPosition2D(sigma_pos), x0, P0)

def build_imm(modes: dict, transition_matrix: list, mode_probs0: list, x0: np.ndarray, P0: np.ndarray) -> InteractingMultipleModel:
    """
    Builds IMM
    """

    cv_filter = KalmanFilter(ConstantVelocity2D(modes['cv']['sigma_a']), LinearPosition2D(sigma_pos), x0, P0)
    ct_plus_filter = KalmanFilter(CoordinatedTurn2D(omega=modes['ct_plus']['omega'], sigma_a=modes['ct_plus']['sigma_a']), LinearPosition2D(sigma_pos), x0, P0)
    ct_minus_filter = KalmanFilter(CoordinatedTurn2D(omega=modes['ct_minus']['omega'], sigma_a=modes['ct_minus']['sigma_a']), LinearPosition2D(sigma_pos), x0, P0)

    # Order needs to be cv, ct_plus, ct_minus
    filters = [cv_filter, ct_plus_filter, ct_minus_filter]
    
    return InteractingMultipleModel(filters, np.array(transition_matrix), np.array(mode_probs0))


# Trial function
def run_trial(seed: int) -> dict:

    rng = np.random.default_rng(seed)  # For reproducibility

    # Make results dictionary
    results = {
                m: {
                    r: {metric: [] for metric in metrics} 
                    for r in regimes
                } 
                for m in methods
            }
    for trajectory in trajectories:

        T = len(trajectory)
        z_seq = trajectory[:, :2] + rng.normal(0, sigma_pos, size=(T, 2))
        x0 = np.concatenate([z_seq[0], np.zeros(2)])

        is_turning, _ = classify_maneuver_segments(trajectory, dt)

        # Create CV filter, run on trajectory, and calculate sq_err and nees
        cv_filter = build_cv_filter(config_imm["cv_baseline"]["sigma_a"], x0, P0)

        x_est_seq_cv, nees_seq_cv = run_filter_on_trajectory(cv_filter, trajectory, z_seq, dt)

        sq_err_cv = np.sum((x_est_seq_cv[:, :2] - trajectory[:, :2])**2, axis=1)

        results["CV"]["overall"]["sq_err"].append(sq_err_cv)
        results["CV"]["overall"]["nees"].append(nees_seq_cv)

        results["CV"]["straight"]["sq_err"].append(sq_err_cv[~is_turning])
        results["CV"]["straight"]["nees"].append(nees_seq_cv[~is_turning])

        results["CV"]["turning"]["sq_err"].append(sq_err_cv[is_turning])
        results["CV"]["turning"]["nees"].append(nees_seq_cv[is_turning])


        # Create IMM, run on trajectory, and calculate sq_err and nees
        imm = build_imm(config_imm["modes"], config_imm["transition_matrix"], config_imm["mode_probs0"], x0, P0)
        x_est_seq_imm, nees_seq_imm = run_filter_on_trajectory(imm, trajectory, z_seq, dt)

        sq_err_imm = np.sum((x_est_seq_imm[:, :2] - trajectory[:, :2])**2, axis=1)

        results["IMM"]["overall"]["sq_err"].append(sq_err_imm)
        results["IMM"]["overall"]["nees"].append(nees_seq_imm)

        results["IMM"]["straight"]["sq_err"].append(sq_err_imm[~is_turning])
        results["IMM"]["straight"]["nees"].append(nees_seq_imm[~is_turning])

        results["IMM"]["turning"]["sq_err"].append(sq_err_imm[is_turning])
        results["IMM"]["turning"]["nees"].append(nees_seq_imm[is_turning])

        # Create ResidualDynamicsKF, run on trajectory, and calculate sq_err and nees

        z_seq_t = torch.tensor(z_seq, dtype=torch.float32).unsqueeze(0) # (1, T, 2)
        x0_t = torch.tensor(x0, dtype=torch.float32).unsqueeze(0) # (1, 4)
        P0_t = torch.tensor(P0, dtype=torch.float32).unsqueeze(0) # (1, 4, 4)
        h0_t = torch.zeros(1, checkpoint_config['dynamics_net']['hidden_dim']) # Zero initial hidden state

        with torch.no_grad():
            x_est_seq_rd, P_est_seq_rd, h_est_seq_rd = model(z_seq_t, x0_t, P0_t, h0_t)

        x_est_seq_rd = x_est_seq_rd.squeeze(0).numpy() # (T, 4)
        P_est_seq_rd = P_est_seq_rd.squeeze(0).numpy() # (T, 4, 4)

        nees_seq_rd = np.zeros(T)

        for t in range(T):
            error = trajectory[t] - x_est_seq_rd[t]
            nees_seq_rd[t] = error @ linalg.solve(P_est_seq_rd[t], error)

        sq_err_rd = np.sum((x_est_seq_rd[:, :2] - trajectory[:, :2])**2, axis=1)

        results["ResidualDynamicsKF"]["overall"]["sq_err"].append(sq_err_rd)
        results["ResidualDynamicsKF"]["overall"]["nees"].append(nees_seq_rd)

        results["ResidualDynamicsKF"]["straight"]["sq_err"].append(sq_err_rd[~is_turning])
        results["ResidualDynamicsKF"]["straight"]["nees"].append(nees_seq_rd[~is_turning])

        results["ResidualDynamicsKF"]["turning"]["sq_err"].append(sq_err_rd[is_turning])
        results["ResidualDynamicsKF"]["turning"]["nees"].append(nees_seq_rd[is_turning])

    return results

# Make overall results dictionary
overall_results = {
                    m: {
                        r: {metric: [] for metric in metrics} 
                        for r in regimes
                    } 
                    for m in methods
                }

# Run MC simulation
for seed in mc_seeds:

    trial_results = run_trial(seed)

    # Add to overall results dictionary
    for m in methods:
        for r in regimes:
            for metric in metrics:
                overall_results[m][r][metric].extend(trial_results[m][r][metric])

# Final results table
stats = {m: {} for m in methods}

for m in methods:
    print(f"{m}")
    for r in regimes:
        rmse = np.sqrt(np.mean(np.concatenate(overall_results[m][r]["sq_err"])))
        print(f"{r} RMSE: {rmse}")

        nees = np.mean(np.concatenate(overall_results[m][r]["nees"]))
        print(f"{r} mean NEES: {nees}")

        stats[m][r] = {"rmse": rmse, "mean_nees": nees}

# RMSE Plot
x = np.arange(len(regimes))
width = 0.25
regime_labels = [r.capitalize() for r in regimes]

plt.figure(figsize=(12, 6))
for i, m in enumerate(methods):
    values_1 = [stats[m][r]["rmse"] for r in regimes]
    plt.bar(x + (i - 1) * width, values_1, width=width, label=m)

plt.xticks(x, regime_labels)
plt.xlabel("Regimes")
plt.ylabel("RMSE")
plt.title("RMSE by Regime and Method")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/run_maneuver_comparison_1.png")
plt.show()


# NEES Plot
plt.figure(figsize=(12, 6))
for i, m in enumerate(methods):
    values_2 = [stats[m][r]["mean_nees"] for r in regimes]
    plt.bar(x + (i - 1) * width, values_2, width=width, label=m)

plt.axhline(y=4, color="black", linestyle=":", label="Ideal NEES (state dim)")
plt.xticks(x, regime_labels)
plt.xlabel("Regimes")
plt.ylabel("NEES")
plt.title("NEES by Regime and Method")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/run_maneuver_comparison_2.png")
plt.show()


    










