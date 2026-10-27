from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
import matplotlib.pyplot as plt
import joblib
from fusion.sim.sensors_nl import RadarSensor
from fusion.learned.clutter_tracker import MultiTargetTrackerClutter
from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.models.motion import ConstantVelocity2D
from tracking.eval.metrics import match_tracks_to_truth, count_id_switches
from fusion.eval.metrics import false_track_rate
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.models.radar import RadarRangeBearing

# This script sweeps the classifier-augmented tracker's birth_threshold across several values.
# The goal is to see what threshold provides the lowest optimal ID switches / false-track rate.

# Load classifier data
bundle = joblib.load("data/processed/clutter_classifier.joblib")
classifier = bundle["model"]
n_active_features = len(bundle["feature_names"])

thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

# Extract Config Variables
mc_trials = config['comparison']['mc_trials']
seed = config['seed']

p_detect = config['clutter']['p_detect']
clutter_rate = config['clutter']['clutter_rate']
range_max = config['clutter']['range_max']
bearing_range = config['clutter']['bearing_range']
n_targets = config['clutter']['n_targets']
n_steps = config['clutter']['n_steps']
dt = config['scenario']['dt']
gate_confidence = config['clutter']['gate_confidence']
spawn_region = config['clutter']['spawn_region']
x_min, x_max, y_min, y_max = spawn_region
speed_range = config['clutter']['speed_range']
speed_min, speed_max = speed_range
sigma_range = config['scenario']['sigma_range']
match_gate = 3 * sigma_range # This is 300 meters
sigma_bearing = config['scenario']['sigma_bearing']

# Construct models and sensor
motion_model = ConstantVelocity2D(sigma_a=0.0)
radar_model = RadarRangeBearing(sigma_range=sigma_range, sigma_bearing=sigma_bearing)
radar_sensor = RadarSensor(radar_model=radar_model, p_detect=p_detect, clutter_rate=clutter_rate, range_max=range_max, bearing_range=bearing_range)


# To collect data from multiple trials (Do type hints after)
def run_trial(seed: int) -> dict[str, dict[str, int | float]]:
    rng = np.random.default_rng(seed)  # For reproducibility

    # Create fresh multi-target scenario
    trajectories = []
    
    for _ in range(n_targets):
        speed = rng.uniform(speed_min, speed_max)
        heading = rng.uniform(-np.pi, np.pi)

        px = rng.uniform(x_min, x_max)
        py = rng.uniform(y_min, y_max)
        vx = speed * np.cos(heading)
        vy = speed * np.sin(heading)
        
        x0 = np.array([px, py, vx, vy])

        trajectory = constant_velocity_trajectory(x0, dt, n_steps)
        trajectories.append(trajectory)

    trajectories = np.asarray(trajectories)

    # EKFs with cluttter classifier (for each threshold) and EKF without
    baseline_tracker = MultiTargetTrackerClutter(motion_model, radar_model, gate_confidence=gate_confidence)
    classifier_trackers = {
        threshold: MultiTargetTrackerClutter(
            motion_model,
            radar_model,
            classifier=classifier,
            n_active_features=n_active_features,
            gate_confidence=gate_confidence,
            birth_threshold=threshold)
        for threshold in thresholds}

    # Assignment log and confirmed tracks lists
    assignment_log_baseline = []
    confirmed_track_ids_per_frame_baseline = []

    assignment_logs = {threshold: [] for threshold in thresholds}

    confirmed_track_ids_per_frame = {threshold: [] for threshold in thresholds}
        

    # For each frame collect assigment log and confirmed tracks
    for t in range(n_steps):
        
        true_states = trajectories[:, t]
        
        true_positions_this_frame = {truth_id: trajectories[truth_id, t, :2] for truth_id in range(n_targets)}

        detections = radar_sensor.observe(true_states, t*dt, rng)

        baseline_result = baseline_tracker.step(detections, dt, t*dt)

        assignment_log_baseline.append(match_tracks_to_truth(baseline_result.snapshots, true_positions_this_frame, match_gate))
        confirmed_track_ids_per_frame_baseline.append({s.track_id for s in baseline_result.snapshots if s.confirmed})

        for threshold, tracker in classifier_trackers.items():
            result = tracker.step(detections, dt, t*dt)
            assignment_logs[threshold].append(match_tracks_to_truth(result.snapshots, true_positions_this_frame, match_gate))
            confirmed_track_ids_per_frame[threshold].append({s.track_id for s in result.snapshots if s.confirmed})

    # Calculate id switches and false track rates for each classifier
    baseline_id_switches = count_id_switches(assignment_log_baseline, list(range(n_targets)))
    baseline_false_track_rate = false_track_rate(assignment_log_baseline, confirmed_track_ids_per_frame_baseline)
    baseline_stats = {"id_switches": baseline_id_switches, "false_track_rate": baseline_false_track_rate}
    
    classifier_stats = {}

    for threshold in thresholds:
        classifier_stats[threshold] = {
            "id_switches": count_id_switches(assignment_logs[threshold], list(range(n_targets)),),
            "false_track_rate": false_track_rate(assignment_logs[threshold], confirmed_track_ids_per_frame[threshold],)
        }

    return {"baseline": baseline_stats, **classifier_stats,}


# Store results across Monte Carlo trials
all_results = {"baseline": {"id_switches": [], "false_track_rate": []}}

for threshold in thresholds:
    all_results[threshold] = {"id_switches": [], "false_track_rate": []}

# Run the simulation
for trial in range(mc_trials):
    trial_result = run_trial(trial)
    for tracker_name, stats in trial_result.items():
        all_results[tracker_name]["id_switches"].append(stats["id_switches"])
        all_results[tracker_name]["false_track_rate"].append(stats["false_track_rate"])

# Find and report mean stats 
print(f"{'Condition':<15}{'ID Switches':>14}{'False Track Rate':>20}")
for name in ["baseline"] + thresholds:
    label = "Baseline" if name == "baseline" else f"Threshold {name:.1f}"
    print(f"{label:<15}{np.mean(all_results[name]['id_switches']):>14.3f}{np.mean(all_results[name]['false_track_rate']):>20.3f}")

# Plot average ID switches across classifier thresholds
mean_id_switches = [np.mean(all_results[threshold]["id_switches"]) for threshold in thresholds]

baseline_mean_id_switches = np.mean(all_results["baseline"]["id_switches"])

plt.figure(figsize=(9, 5))
plt.plot(
    thresholds,
    mean_id_switches,
    marker="o",
    label="Classifier",
)
plt.axhline(
    baseline_mean_id_switches,
    linestyle="--",
    label="Baseline",
)
plt.xlabel("Birth Threshold")
plt.ylabel("Average ID Switches")
plt.title("ID Switches vs. Classifier Birth Threshold")
plt.xticks(thresholds)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("figures/sweep_birth_threshold_1.png")
plt.show()

# Plot average false track rate across classifier thresholds
mean_false_track_rates = [np.mean(all_results[threshold]["false_track_rate"]) for threshold in thresholds]

baseline_mean_false_track_rate = np.mean(all_results["baseline"]["false_track_rate"])

plt.figure(figsize=(9, 5))
plt.plot(
    thresholds,
    mean_false_track_rates,
    marker="o",
    label="Classifier",
)
plt.axhline(
    baseline_mean_false_track_rate,
    linestyle="--",
    label="Baseline",
)
plt.xlabel("Birth Threshold")
plt.ylabel("Average False Track Rate")
plt.title("False Track Rate vs. Classifier Birth Threshold")
plt.xticks(thresholds)
plt.ylim(0, 1)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig("figures/sweep_birth_threshold_2.png")
plt.show()









