from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "training.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)
import joblib

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from tracking.models.motion import ConstantVelocity2D
from fusion.models.radar import RadarRangeBearing
from fusion.sim.sensors_nl import RadarSensor
from fusion.learned.clutter import build_clutter_dataset, check_calibration
import matplotlib.pyplot as plt

# Set seed
rng = np.random.default_rng(config['seed'])

# Create models and sensor

motion_model = ConstantVelocity2D(sigma_a=0.0)
radar_model = RadarRangeBearing(sigma_range=config['scenario']['sigma_range'], sigma_bearing=config['scenario']['sigma_bearing'])

radar_sensor = RadarSensor(radar_model=radar_model,
                     p_detect=config['clutter']['p_detect'],
                     clutter_rate=config['clutter']['clutter_rate'],
                     range_max=config['clutter']['range_max'],
                     bearing_range=config['clutter']['bearing_range'])

# Build train/val clutter dataset

train_X, train_y = build_clutter_dataset(radar_sensor=radar_sensor,
                             motion_model=motion_model,
                             n_targets=config['clutter']['n_targets'],
                             n_scenarios=config['clutter']['n_scenarios_train'],
                             n_steps=config['clutter']['n_steps'],
                             dt=config['scenario']['dt'],
                             gate_confidence=config['clutter']['gate_confidence'],
                             spawn_region=config['clutter']['spawn_region'],
                             speed_range=config['clutter']['speed_range'],
                             rng=rng)

val_X, val_y = build_clutter_dataset(radar_sensor=radar_sensor,
                             motion_model=motion_model,
                             n_targets=config['clutter']['n_targets'],
                             n_scenarios=config['clutter']['n_scenarios_val'],
                             n_steps=config['clutter']['n_steps'],
                             dt=config['scenario']['dt'],
                             gate_confidence=config['clutter']['gate_confidence'],
                             spawn_region=config['clutter']['spawn_region'],
                             speed_range=config['clutter']['speed_range'],
                             rng=rng)

# Check label balance (prior to training)
train_counts = np.bincount(train_y)
val_counts = np.bincount(val_y)

print("Label Balance Diagnostics")
print(f"Training samples: {len(train_y):,}")
print(f"Validation samples: {len(val_y):,}")

print("Training label balance:")
print(f"Class 0: {train_counts[0]:,} ({train_counts[0] / len(train_y):.2%})")
print(f"Class 1: {train_counts[1]:,} ({train_counts[1] / len(train_y):.2%})")

print("Validation label balance:")
print(f"Class 0: {val_counts[0]:,} ({val_counts[0] / len(val_y):.2%})")
print(f"Class 1: {val_counts[1]:,} ({val_counts[1] / len(val_y):.2%})")


# Create classifier and fit (only three features until I implement SNR)
clf = HistGradientBoostingClassifier(random_state=config['seed'])
clf.fit(train_X[:, :3], train_y)
val_probs = clf.predict_proba(val_X[:, :3])[:, 1]

# Save model and metadata (no SNR yet)
model_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "clutter_classifier.joblib"
joblib.dump({
    "model": clf,
    "feature_names": ["min_distance_sq", "n_gated", "local_density"],
    "config": config["clutter"],
}, model_path)

# Get diagnostics (AUC for accuracy)
auc = roc_auc_score(val_y, val_probs)

calibration_diagnostics = check_calibration(val_y, val_probs)
brier_score = calibration_diagnostics["brier_score"]
bin_counts = calibration_diagnostics["bin_counts"]
prob_pred = calibration_diagnostics["prob_pred"]
prob_true = calibration_diagnostics["prob_true"]
bin_edges = calibration_diagnostics["bin_edges"]


# Print diagnostics and table for bin counts
print("Training Diagnostics")
print(f"ROC AUC: {auc:.4f}")
print(f"Brier score: {brier_score:.4f}")

print("Calibration by bin")
for b in range(len(bin_counts)):
    print(
        f"Bin {b}: "
        f"count={bin_counts[b]}, "
        f"prob_pred={prob_pred[b]:.4f}, "
        f"prob_true={prob_true[b]:.4f}"
    )

# Plot Reliability Diagram
plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
plt.plot(prob_pred, prob_true, "o-", label="Model")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Observed Positive Fraction")
plt.title("Radar Clutter Classifier Reliability Diagram")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("figures/train_clutter_classifier.png")
plt.show()

