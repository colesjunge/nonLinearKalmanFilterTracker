from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
from fusion.sim.maneuvers import waypoint_trajectory
from fusion.sim.noise import contaminated_gaussian_noise
from fusion.learned.datasets import extract_windows
from fusion.models.radar import RadarRangeBearing


# This file generates manuevering datasets with contaminated noise for full pipeline model

# Set up config variables
seed = config['seed']
n_trajectories = config['scenario']['n_trajectories']
dt = config['scenario']['dt']
n_steps = config['scenario']['n_steps']

min_x = config['scenario']['min_x']
max_x = config['scenario']['max_x']
min_y = config['scenario']['min_y']
max_y = config['scenario']['max_y']

turn_prob = config['maneuver']['turn_prob']
leg_duration_range = config['maneuver']['leg_duration_range']
omega_range = config['maneuver']['omega_range']

sigma_range = config['scenario']['sigma_range']
sigma_bearing = config['scenario']['sigma_bearing']
epsilon = config['scenario']['epsilon']
kappa = config['scenario']['kappa']
window = config['dataset']['window']

rng = np.random.default_rng(seed)
radar_model = RadarRangeBearing(sigma_range=sigma_range , sigma_bearing=sigma_bearing)

# Collect all trajectories and windows
all_windows = []
all_trajectories = []

for traj in range(n_trajectories):

    # Random start state
    x = rng.uniform(min_x, max_x)
    y = rng.uniform(min_y, max_y)

    speed = rng.uniform(5, 30)
    heading = rng.uniform(0, 2 * np.pi)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    x0_true = np.array([x, y, vx, vy])

    # Ground truth trajectory
    trajectory = waypoint_trajectory(x0_true, dt, n_steps, rng, turn_prob, leg_duration_range, omega_range)

    z_sequence = []
    positions = []

    for i in range(1, n_steps):
        z_sequence.append(radar_model.h(trajectory[i]) + contaminated_gaussian_noise(rng, radar_model.R(), epsilon, kappa))
        positions.append(trajectory[i][:2])

    z_sequence = np.array(z_sequence)
    positions = np.array(positions)


    windows, _ = extract_windows(z_sequence, positions, window)

    all_windows.append(windows)
    all_trajectories.append(trajectory)

all_windows = np.array(all_windows)
all_trajectories = np.array(all_trajectories)

# Save to output directory
output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "full_pipeline_dataset.npz"
np.savez(output_path, trajectories=all_trajectories, windows=all_windows)
print(f"Saved {all_trajectories.shape} to {output_path}")


