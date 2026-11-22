from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
from tracking.sim.trajectories import constant_velocity_trajectory
from fusion.models.radar import RadarRangeBearing
from fusion.sim.noise import contaminated_gaussian_noise
from fusion.learned.datasets import extract_windows

# This file generates constant velocity datasets with the radarRangeBearing model with contaminated noise

# Set up config variables
seed = config['seed']
sigma_range = config['scenario']['sigma_range']
sigma_bearing = config['scenario']['sigma_bearing']
n_trajectories = config['scenario']['n_trajectories']
dt = config['scenario']['dt']
n_steps = config['scenario']['n_steps']
epsilon = config['scenario']['epsilon']
kappa = config['scenario']['kappa']
window = config['dataset']['window']

rng = np.random.default_rng(seed)
radar_model = RadarRangeBearing(sigma_range=sigma_range , sigma_bearing=sigma_bearing)

# For all randomly generated trajectories collect windows and targets
all_windows = []
all_targets = []

for traj in range(n_trajectories):

    # Random start state
    x = rng.uniform(-100, 100)
    y = rng.uniform(-100, 100)

    speed = rng.uniform(5, 30)
    heading = rng.uniform(0, 2 * np.pi)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    x0_true = np.array([x, y, vx, vy])

    # Ground truth trajectory
    trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

    z_sequence = []
    positions = []

    for i in range(1, n_steps):
        z_sequence.append(radar_model.h(trajectory[i]) + contaminated_gaussian_noise(rng, radar_model.R(), epsilon, kappa))
        positions.append(trajectory[i][:2])

    z_sequence = np.array(z_sequence)
    positions = np.array(positions)


    windows, targets = extract_windows(z_sequence, positions, window)

    all_windows.append(windows)
    all_targets.append(targets)

all_windows = np.array(all_windows)
all_targets = np.array(all_targets)

# Save to output directory
output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "denoiser_dataset.npz"
np.savez(output_path, windows=all_windows, targets=all_targets)
print(f"Saved {all_windows.shape} to {output_path}")


