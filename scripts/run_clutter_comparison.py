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

# This script compares a pure-kinematic-gate multi-target tracker against the same tracker augmented with the trained clutter classifier's birth-gating
# The metrics are ID switches and false-track rate

# Load classifier data
bundle = joblib.load("data/processed/clutter_classifier.joblib")
classifier = bundle["model"]
n_active_features = len(bundle["feature_names"])

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

    # EKF with cluttter classifier and EKF without (birth_threshold set to .3 based on sweep_birth_threshold.py results)
    baseline_tracker = MultiTargetTrackerClutter(motion_model, radar_model, gate_confidence=gate_confidence)
    classifier_tracker = MultiTargetTrackerClutter(motion_model, radar_model, classifier=classifier, n_active_features=n_active_features, gate_confidence=gate_confidence, birth_threshold=.3)

    # Assignment log and confirmed tracks lists
    assignment_log_baseline = []
    assignment_log_classifier = []
    confirmed_track_ids_per_frame_baseline = []
    confirmed_track_ids_per_frame_classifier = []

    # For each frame collect assigment log and confirmed tracks
    for t in range(n_steps):
        
        true_states = trajectories[:, t]

        detections = radar_sensor.observe(true_states, t*dt, rng)

        baseline_result = baseline_tracker.step(detections, dt, t*dt)
        classifier_result = classifier_tracker.step(detections, dt, t*dt)

        true_positions_this_frame = {truth_id: trajectories[truth_id, t, :2] for truth_id in range(n_targets)}

        assignment_log_baseline.append(match_tracks_to_truth(baseline_result.snapshots, true_positions_this_frame, match_gate))
        assignment_log_classifier.append(match_tracks_to_truth(classifier_result.snapshots, true_positions_this_frame, match_gate))

        confirmed_track_ids_per_frame_baseline.append({s.track_id for s in baseline_result.snapshots if s.confirmed})
        confirmed_track_ids_per_frame_classifier.append({s.track_id for s in classifier_result.snapshots if s.confirmed})

    # Calculate id switches and false track rates for each classifier
    baseline_id_switches = count_id_switches(assignment_log_baseline, list(range(n_targets)))
    baseline_false_track_rate = false_track_rate(assignment_log_baseline, confirmed_track_ids_per_frame_baseline)

    classifier_id_switches = count_id_switches(assignment_log_classifier, list(range(n_targets)))
    classifier_false_track_rate = false_track_rate(assignment_log_classifier, confirmed_track_ids_per_frame_classifier)

    baseline_stats = {"id_switches": baseline_id_switches, "false_track_rate": baseline_false_track_rate}
    classifier_stats = {"id_switches": classifier_id_switches, "false_track_rate": classifier_false_track_rate}

    return {"baseline": baseline_stats, "classifier": classifier_stats}


all_results = {
    "baseline": {"id_switches": [], "false_track_rate": []},
    "classifier": {"id_switches": [], "false_track_rate": []},
}

# Run the simulation
for trial in range(mc_trials):
    trial_result = run_trial(trial)
    for tracker_name, stats in trial_result.items():
        all_results[tracker_name]["id_switches"].append(stats["id_switches"])
        all_results[tracker_name]["false_track_rate"].append(stats["false_track_rate"])

# Find and report mean stats 
for tracker_name, stats in all_results.items():
    print(f"Average amount of id_switches ({tracker_name}): {np.mean(stats['id_switches'])}")
    print(f"Average false_track_rate ({tracker_name}): {np.mean(stats['false_track_rate'])}")

# Plot average ID switches
tracker_names = list(all_results.keys())
labels = ["Baseline", "Classifier"]

mean_id_switches = [
    np.mean(all_results[name]["id_switches"])
    for name in tracker_names
]

plt.figure(figsize=(8, 5))
plt.bar(labels, mean_id_switches)
plt.xlabel("Tracker")
plt.ylabel("Average ID Switches")
plt.title("Average ID Switches for Each Scenario")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("figures/run_clutter_comparison_1.png")
plt.show()

# Plot average false track rates
tracker_names = list(all_results.keys())
labels = ["Baseline", "Classifier"]

mean_false_track_rates = [
    np.mean(all_results[name]["false_track_rate"])
    for name in tracker_names
]

plt.figure(figsize=(8, 5))
plt.bar(labels, mean_false_track_rates)
plt.xlabel("Tracker")
plt.ylabel("Average False Track Rate")
plt.title("Average False Track Rate for Each Scenario")
plt.grid(axis="y")
plt.tight_layout()
plt.savefig("figures/run_clutter_comparison_2.png")
plt.show()









