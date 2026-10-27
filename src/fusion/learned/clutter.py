import numpy as np

from tracking.types import Detection
from tracking.tracker.track import Track
from tracking.association.gating import mahalanobis_sq, chi2_gate_threshold
from tracking.models.motion import MotionModel
from fusion.sim.sensors_nl import RadarSensor
from fusion.filters.ekf import ExtendedKalmanFilter
from tracking.sim.trajectories import constant_velocity_trajectory
from fusion.models.radar import RadarRangeBearing

N_FEATURES = 4

def init_track_from_detection(det: Detection, radar_model: RadarRangeBearing, motion_model: MotionModel, track_id: int, timestamp: float) -> Track:
    """ Builds a new Track from an unmatched detection
    Inverts (range, bearing) to (px, py)
    Does not assign an ID or register the track anywhere
    In:  det, the unmatched Detection (range, bearing) to spawn a track from;
         radar_model, the nonlinear measurement model used to build the track's EKF;
         motion_model, the motion model used to build the track's EKF;
         track_id, the ID the caller has already decided for this new track;
         timestamp, absolute time the track is created;
    Out: Track, wrapping a fresh ExtendedKalmanFilter seeded from the inverted position
    """

    # Determine px, py from range and bearing
    px = det.z[0]*np.cos(det.z[1])
    py = det.z[0]*np.sin(det.z[1])

    x0 = np.array([px, py, 0, 0])

    P0 = np.array([[10, 0, 0, 0], 
            [0, 12, 0, 0], 
            [0, 0, 34, 0], 
            [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)

    ekf = ExtendedKalmanFilter(motion_model, radar_model, x0, P0)

    track = Track(track_id, ekf, timestamp)

    return track

def compute_features(detection: Detection, detections_this_frame: list[Detection], tracks: list[Track], gate: float) -> np.ndarray:
    """ Per-detection feature engineering for the clutter classifier
    In:  detection, the single Detection to compute features for;
        detections_this_frame, all Detections (real + clutter) in the current frame;
        tracks, list[Track], all currently active tracks;
        gate, chi-square gate threshold from chi2_gate_threshold
    Out: feature vector (N_FEATURES,) [squared Mahalanobis distance to nearest track,
                                        count of tracks this detection gates against,
                                        local detection density,
                                        SNR if present in meta, else NaN]
    """

    # Compute squared Mahalanobis distance for each track
    distances = []
    for track in tracks:
        y, S = track.filter.innovation(detection.z)
        d_sq = mahalanobis_sq(y, S)
        distances.append(d_sq)

    # Nearest-track distance and number of tracks gated against
    if distances:
        min_distance = min(distances)
        n_gated = sum(d <= gate for d in distances)
    else:
        # Sentinel when no tracks exist
        min_distance = np.inf
        n_gated = 0

    # Local density (number of other detections in the current frame)
    local_density = max(0, len(detections_this_frame) - 1)

    # SNR if present in meta
    snr = detection.meta.get("snr", np.nan)

    return np.array([min_distance, n_gated, local_density, snr], dtype=float)

    
    

def build_clutter_dataset(radar_sensor: RadarSensor,
                        motion_model: MotionModel,
                        n_targets: int,
                        n_scenarios: int,
                        n_steps: int,
                        dt: float,
                        gate_confidence: float,
                        spawn_region: tuple,
                        speed_range: tuple,
                        rng: np.random.Generator,
                        ) -> tuple[np.ndarray, np.ndarray]:
    """ Generates labeled clutter-classifier training data by running n_scenarios independent
    multi-target scenarios, each with n_targets ground-truth-tracked targets over n_steps frames.
    In:  radar_sensor, generates this frame's real and clutter detections;
         motion_model, used to build each target's tracking filter;
         n_targets, simultaneous targets per scenario;
         n_scenarios, independent scenario repeats (fresh trajectories/tracks each time);
         n_steps, frames per scenario;
         dt, seconds per frame;
         gate_confidence, passed to chi2_gate_threshold;
         spawn_region, (x_min, x_max, y_min, y_max) bounds for initial target position;
         speed_range, (speed_min, speed_max) bounds for initial target speed;
         rng, for reproducibility;
    Out: X (n_detections, N_FEATURES), y (n_detections,) (labels from meta["clutter"])
    """

    gate = chi2_gate_threshold(radar_sensor.dim_z, gate_confidence)

    all_features = []
    all_labels = []

    x_min, x_max, y_min, y_max = spawn_region
    speed_min, speed_max = speed_range

    for _ in range(n_scenarios):

        # Generate independent ground-truth trajectories
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

        tracks = {}

        # Simulate detections and collect features
        for t in range(n_steps):

            for track in tracks.values():
                track.predict(dt)

            true_states = trajectories[:, t, :]

            detections_this_frame = radar_sensor.observe(
                true_states,
                t * dt,
                rng,
            )

            # Include BOTH target detections and clutter
            frame_tracks = list(tracks.values())

            for det in detections_this_frame:
                features = compute_features(
                    det,
                    detections_this_frame,
                    frame_tracks,
                    gate,
                )

                is_clutter = det.meta.get("clutter", False)

                all_features.append(features)
                all_labels.append(int(not is_clutter))

                # Ground-truth association is used only for track updates (conditional on if target already has a track)
                if not is_clutter:
                    target_id = det.meta["target_id"]
                    if target_id not in tracks:
                        tracks[target_id] = init_track_from_detection(det, radar_sensor.radar_model, motion_model, target_id, t*dt)
                    else:
                        tracks[target_id].update(det)
    # Convert collected samples into arrays
    if not all_features:
        return np.empty((0, N_FEATURES), dtype=float), np.empty(0, dtype=int)

    X = np.asarray(all_features, dtype=float)
    y = np.asarray(all_labels, dtype=int)

    return X, y
    
def check_calibration(y_true: np.ndarray, probs: np.ndarray, n_bins: int | None = None) -> dict:
    """Explicit calibration check (no plotting).
    In:
        y_true: (n,) binary ground-truth labels
        probs: (n,) predicted probabilities of target origin
        n_bins: number of calibration bins

    Out:
        Calibration diagnostics dictionary
            bin_edges: bin boundaries;
            bin_counts: number of samples per bin;
            prob_pred: mean predicted probability per bin;
            prob_true: empirical positive rate per bin;
            brier_score: overall Brier score
    """

    y_true = np.asarray(y_true)
    probs = np.asarray(probs, dtype=float)

    # Choose the number of bins (roughly one per 50 samples)
    if n_bins is None:
        n_bins = max(5, min(20, len(y_true) // 50))

    # Quantile-based bin edges
    bin_edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))

    # Assign every prediction to a bin
    bin_idx = np.digitize(probs, bin_edges[1:-1])
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Count samples in each bin
    bin_counts = np.bincount(bin_idx, minlength=n_bins)

    # Compute mean predicted probability and empirical positive rate
    prob_pred = np.full(n_bins, np.nan)
    prob_true = np.full(n_bins, np.nan)

    for b in range(n_bins):
        mask = bin_idx == b

        if bin_counts[b] > 0:
            prob_pred[b] = probs[mask].mean()
            prob_true[b] = y_true[mask].mean()

    # Overall Brier score
    brier_score = np.mean((probs - y_true) ** 2)

    return {
        "bin_edges": bin_edges,
        "bin_counts": bin_counts,
        "prob_pred": prob_pred,
        "prob_true": prob_true,
        "brier_score": brier_score,
    }

def build_cost_matrix_with_classifier(tracks: list[Track], detections: list[Detection], gate: float, classifier, n_active_features: int) -> np.ndarray:
    """ Create cost matrix using track and detection pairs using mahalanobis distance and cutoffs
    Also incorporates classifier predictions as an additional term
    Does not overide kinematic predictions, just reweights based on classifier confidence

    In:  tracks, list[Track] of length M, the currently active tracks;
         detections, list[Detection] of length N, this frame's raw detections (real and clutter);
         gate, chi-square gate threshold from chi2_gate_threshold;
         classifier, a trained scikit-learn-style classifier with .predict_proba(), fit on
                     compute_features()'s output (sliced to n_active_features columns);
         n_active_features, number of leading feature columns the classifier was trained on;
    Out: cost (M, N)
    """

    M = len(tracks)
    N = len(detections)
    cost = np.zeros((M, N), dtype=float)

    features = []

    # Compute features, feature matrix and create probabilities
    for j in range(N):
        features.append(compute_features(detections[j], detections, tracks, gate))

    features = np.vstack(features)
    features = features[:, :n_active_features]

    probabilities = classifier.predict_proba(features)[:, 1]

    # Assign costs
    for i in range(M):
        for j in range(N):
            y, S = tracks[i].filter.innovation(detections[j].z)
            d_sq = mahalanobis_sq(y, S)
            if d_sq <= gate:   
                cost[i, j] = d_sq - 2 * np.log(np.clip(probabilities[j], 1e-12, 1 - 1e-12))
            else:
                cost[i, j] = np.inf
    
    return cost




    
