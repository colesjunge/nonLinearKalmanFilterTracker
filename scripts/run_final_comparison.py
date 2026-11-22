from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "final_comparison.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import json
import joblib
import numpy as np
import torch
from scipy import linalg

from fusion.eval.scenarios import build_benign_scenario, build_maneuvering_scenario, build_contaminated_scenario, build_bearing_only_scenario
from fusion.eval.filters import build_benign_filter, build_maneuvering_filter, build_contaminated_filter, build_bearing_only_filter, build_imm_filter
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.ukf import UnscentedKalmanFilter
from fusion.fuse.naive import fuse_naive
from fusion.fuse.covariance_intersection import covariance_intersection

from fusion.learned.clutter_tracker import MultiTargetTrackerClutter
from fusion.sim.sensors_nl import RadarSensor
from fusion.models.radar import RadarRangeBearing
from tracking.models.motion import ConstantVelocity2D
from tracking.sim.trajectories import constant_velocity_trajectory
from tracking.eval.metrics import match_tracks_to_truth, count_id_switches, nees
from fusion.eval.metrics import false_track_rate

from tracking.models.measurement import LinearPosition2D
from tracking.filters.kalman import KalmanFilter
from fusion.models.adaptive_position import AdaptivePosition2D
from fusion.sim.noise import contaminated_gaussian_noise
from fusion.learned.checkpoints import load_checkpoint
from fusion.learned.datasets import load_denoiser_dataset, extract_windows
from fusion.learned.denoiser import build_denoiser
from fusion.learned.covariance import build_rollout_covariance_model, cov_from_chol

from fusion.learned.datasets import load_maneuver_dataset
from fusion.learned.residual_dynamics_kf import build_residual_dynamics_kf

from fusion.learned.full_pipeline import build_full_pipeline_model, build_full_pipeline_bearing_model
from fusion.sim.maneuvers import waypoint_trajectory

from fusion.models.eoir import EOIRBearingOnly

from comparison_common_fn import run_comparison_for_cell, run_comparison_for_sequential, run_comparison_for_track_level

# Functions to create rows below

def run_benign_rows(config: dict, mc_seeds: list[int]) -> dict[str, dict]:
    """ Creates results for benign rows for EKF and UKF
    In: config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    Out: row_dict, {"EKF": {"rmse":..,"mean_nees":..}, "UKF": {...}}
    """

    row_dict = {}

    # Cycle through methods
    for method in config['scenarios']['benign']['methods']:

        # Get tuned sigma_a
        sigma_a = config['tuned']['benign'][method]

        # Build method filter
        if method == "EKF":
            filter_builder_fn = lambda traj, z: build_benign_filter(sigma_a, traj, z, ExtendedKalmanFilter, config)
        elif method == "UKF":
            filter_builder_fn = lambda traj, z: build_benign_filter(sigma_a, traj, z, UnscentedKalmanFilter, config)

        # Get stats for said method
        stat_dict = run_comparison_for_cell(build_benign_scenario, filter_builder_fn, config, mc_seeds)

        # Add stats to row_dict
        row_dict[method] = stat_dict

    return row_dict


def run_maneuvering_rows(config: dict, mc_seeds: list[int]) -> dict[str, dict]:
    """ Creates results for maneuvering rows for EKF, UKF, IMM
    In: config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    Out: row_dict, {"EKF": {"straight":{...},"turning":{...},"overall":{...}}, "UKF": {...}, "IMM": {...}}
    """

    row_dict = {}

    # Cycle through methods
    for method in config['scenarios']['maneuvering']['methods']:

        # Get tuned sigma_a (IMM are hardcoded in build_imm_filter)
        if method != "IMM":
            sigma_a = config['tuned']['maneuvering'][method]

        # Build method filter
        if method == "EKF":
            filter_builder_fn = lambda traj, z: build_maneuvering_filter(sigma_a, traj, z, ExtendedKalmanFilter, config)
        elif method == "UKF":
            filter_builder_fn = lambda traj, z: build_maneuvering_filter(sigma_a, traj, z, UnscentedKalmanFilter, config)
        elif method == "IMM":
            filter_builder_fn = lambda traj, z: build_imm_filter(traj, z, config)

        # Get stats for said method
        stat_dict = run_comparison_for_cell(build_maneuvering_scenario, filter_builder_fn, config, mc_seeds, split_by_regime=True)

        # Add stats to row_dict
        row_dict[method] = stat_dict

    return row_dict

def run_contaminated_rows(config: dict, mc_seeds: list[int]) -> dict[str, dict]:
    """ Creates results for contaminated rows for EKF (UKF cannot complete a rollout under this contamination level)
    In: config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    Out: row_dict, {"EKF": {"rmse":..,"mean_nees":..}}
    """
    
    row_dict = {}
    
    # Cycle through methods
    for method in config['scenarios']['contaminated']['methods']:

        # Get tuned sigma_a
        sigma_a = config['tuned']['contaminated'][method]

        # Build method filter
        if method == "EKF":
            filter_builder_fn = lambda traj, z: build_contaminated_filter(sigma_a, traj, z, ExtendedKalmanFilter, config)
       
        # Get stats for said method and add stats to row_dict
        if method == "UKF":
            row_dict[method] = None
        else:
            stat_dict = run_comparison_for_cell(build_contaminated_scenario, filter_builder_fn, config, mc_seeds)
            row_dict[method] = stat_dict

    return row_dict

def run_bearing_only_rows(config: dict, mc_seeds: list[int]) -> dict[str, dict]:
    """ Creates results for bearing only rows for EKF and UKF
    In: config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    Out: row_dict, {"EKF": {"rmse":..,"mean_nees":..}, "UKF": {...}}
    """

    row_dict = {}
    
    # Cycle through methods
    for method in config['scenarios']['bearing_only']['methods']:

        # Get tuned sigma_a
        sigma_a = config['tuned']['bearing_only'][method]

        # Build method filter
        if method == "EKF":
            filter_builder_fn = lambda traj, z: build_bearing_only_filter(sigma_a, traj, z, ExtendedKalmanFilter, config)
        elif method == "UKF":
            filter_builder_fn = lambda traj, z: build_bearing_only_filter(sigma_a, traj, z, UnscentedKalmanFilter, config)

        # Get stats for said method
        stat_dict = run_comparison_for_cell(build_bearing_only_scenario, filter_builder_fn, config, mc_seeds)

        # Add stats to row_dict
        row_dict[method] = stat_dict

    return row_dict

def run_dual_sensor_fusion_rows(config: dict, mc_seeds: list[int]) -> dict[str, dict]:
    """ Creates results for dual sensor fusion only rows for EKF and UKF (sequential, naive, CI)
    In: config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    Out: row_dict, {"sequential_EKF":{...}, "sequential_UKF":{...}, 
                    "naive_EKF":{...}, "naive_UKF":{...},
                    "ci_EKF":{...}, "ci_UKF":{...}}
    """
    
    row_dict = {}
    
    # Cycle through methods (hardcoded for ease)
    for method in ["EKF", "UKF"]:

        # Build filter_cls
        if method == "EKF":
            filter_cls = ExtendedKalmanFilter 
        elif method == "UKF":
            filter_cls = UnscentedKalmanFilter

        # Add stats to row_dict
        row_dict[f"sequential_{method}"] = run_comparison_for_sequential(filter_cls, config, mc_seeds)
        row_dict[f"naive_{method}"] = run_comparison_for_track_level(filter_cls, fuse_naive, config, mc_seeds)
        row_dict[f"ci_{method}"] = run_comparison_for_track_level(filter_cls, covariance_intersection, config, mc_seeds)
    
    return row_dict

def run_heavy_clutter_rows(config: dict, mc_seeds: list[int]) -> dict[str, dict]:
    """ Creates results for heavy_clutter rows for EKF and UKF
    In: config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    Out: row_dict, {"EKF": {"id_switches":..,"false_track_rate":..}, "UKF": {...}}
    """

    row_dict = {}

    # Load classifier data
    bundle = joblib.load("data/processed/clutter_classifier.joblib")
    classifier = bundle["model"]
    n_active_features = len(bundle["feature_names"])

    # Extract Config Variables
    p_detect = config['scenarios']['heavy_clutter']['p_detect']
    clutter_rate = config['scenarios']['heavy_clutter']['clutter_rate']
    range_max = config['scenarios']['heavy_clutter']['range_max']
    bearing_range = config['scenarios']['heavy_clutter']['bearing_range']
    n_targets = config['scenarios']['heavy_clutter']['n_targets']
    n_steps = config['n_steps']
    dt = config['dt']
    gate_confidence = config['scenarios']['heavy_clutter']['gate_confidence']
    spawn_region = config['scenarios']['heavy_clutter']['spawn_region']
    x_min, x_max, y_min, y_max = spawn_region
    speed_range = config['scenarios']['heavy_clutter']['speed_range']
    speed_min, speed_max = speed_range
    sigma_range = config['scenarios']['heavy_clutter']['sigma_range']
    match_gate = config['scenarios']['heavy_clutter']['match_gate']
    sigma_bearing = config['scenarios']['heavy_clutter']['sigma_bearing']

    # Construct models and sensor
    motion_model = ConstantVelocity2D(sigma_a=0.0)
    radar_model = RadarRangeBearing(sigma_range=sigma_range, sigma_bearing=sigma_bearing)
    radar_sensor = RadarSensor(radar_model=radar_model, p_detect=p_detect, clutter_rate=clutter_rate, range_max=range_max, bearing_range=bearing_range)

    # Cycle through methods
    for method in config['scenarios']['heavy_clutter']['methods']:

        all_id_switches = []
        all_false_track_rates = []

        # Cycle through mc_seeds
        for seed in mc_seeds:
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

            # method with cluttter classifier
            if method == "EKF":
                filter_cls = ExtendedKalmanFilter
            elif method == "UKF":
                filter_cls = UnscentedKalmanFilter

            classifier_tracker = MultiTargetTrackerClutter(motion_model, radar_model, classifier=classifier, n_active_features=n_active_features, gate_confidence=gate_confidence, birth_threshold=.3, filter_cls=filter_cls)

            # Assignment log and confirmed tracks lists
            assignment_log_classifier = []
            confirmed_track_ids_per_frame_classifier = []

            # For each frame collect assigment log and confirmed tracks
            for t in range(n_steps):
                
                true_states = trajectories[:, t]

                detections = radar_sensor.observe(true_states, t*dt, rng)

                classifier_result = classifier_tracker.step(detections, dt, t*dt)

                true_positions_this_frame = {truth_id: trajectories[truth_id, t, :2] for truth_id in range(n_targets)}

                assignment_log_classifier.append(match_tracks_to_truth(classifier_result.snapshots, true_positions_this_frame, match_gate))

                confirmed_track_ids_per_frame_classifier.append({s.track_id for s in classifier_result.snapshots if s.confirmed})

            # Calculate id switches and false track rates for classifier
            classifier_id_switches = count_id_switches(assignment_log_classifier, list(range(n_targets)))
            classifier_false_track_rate = false_track_rate(assignment_log_classifier, confirmed_track_ids_per_frame_classifier)

            all_id_switches.append(classifier_id_switches)
            all_false_track_rates.append(classifier_false_track_rate)
            
        stat_dict = {"id_switches": np.mean(all_id_switches), "false_track_rate": np.mean(all_false_track_rates)}

        # Add stats to row_dict
        row_dict[method] = stat_dict

    return row_dict

def run_denoiser_rows(config: dict, mc_seeds: list[int]) -> dict[str, dict]:
    """ Creates results for denoiser rows for B1, B1_B4
    In: config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    Out: row_dict, {"B1_denoiser": {"rmse":..,"mean_nees":..}, "B1_B4_rollout": {...}}
    """

    # Load denoiser and adaptive denoiser models, norm stats, and config data from prior created dataset (denoiser_checkpoint)
    model, normalization_stats, checkpoint_config = load_checkpoint(path="data/processed/denoiser_checkpoint.pt", build_model_fn=build_denoiser)
    model.eval()

    rollout_model, _, _ = load_checkpoint(path="data/processed/rollout_covariance_checkpoint.pt", build_model_fn=build_rollout_covariance_model)
    rollout_model.eval()

    # Extract checkpoint_config and config variables
    sigma_range = checkpoint_config['scenario']['sigma_range']
    sigma_bearing = checkpoint_config['scenario']['sigma_bearing']
    epsilon = checkpoint_config['scenario']['epsilon']
    kappa = checkpoint_config['scenario']['kappa']
    window = checkpoint_config['dataset']['window']
    dt = checkpoint_config['scenario']['dt']
    n_steps = checkpoint_config['scenario']['n_steps']

    val_fraction = checkpoint_config['dataset']['val_fraction']
    seed = checkpoint_config['seed']
    dim_pos = checkpoint_config['model']['dim_pos']


    batch_size = checkpoint_config['training']['batch_size']

    # Compute sigma_pos empirically
    _, val_dataset, _ = load_denoiser_dataset(path='data/processed/denoiser_dataset.npz', val_fraction=val_fraction, seed=seed)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Shuffling unnecessary for val set

    errors = []
    with torch.no_grad():
        for windows, targets in val_loader:
            predictions = model(windows) # Forward pass
            errors.append(predictions-targets) 

    # Error is (batch_size, dim_pos) to become (n_val_samples, dim_pos)
    all_errors = torch.cat(errors, dim=0)

    # This is done as the original RMSE function would inflate bias if present
    std = all_errors.std(dim=0)
    sigma_pos = torch.sqrt(torch.mean(std ** 2)).item()

    # Build radar model
    radar_model = RadarRangeBearing(sigma_range=sigma_range, sigma_bearing=sigma_bearing)

    # No tuned value so contaminated EKF will have to do (choosing to focus on other aspects of project)
    motion_model = ConstantVelocity2D(sigma_a=config['tuned']['contaminated']['EKF'])

    all_sq_err = {"B1_denoiser": [], "B1_B4_rollout": []}
    all_nees = {"B1_denoiser": [], "B1_B4_rollout": []}

    for seed in mc_seeds:
        rng = np.random.default_rng(seed)  # For reproducibility
        
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
        z_contaminated_history = []

        # Set up arrays
        results = {"B1_denoiser": {"pos": [], "nees": []}, 
                   "B1_B4_rollout": {"pos": [], "nees": []}}
        ground_truth = []
        z_contaminated_history = []
        
        # Learned-model outputs (None until the window is full)
        denoiser_z = None
        rollout_z = None

        for t in range(1, n_steps):
            
            truth = trajectory[t]
            ground_truth.append(truth[:2])

            # Contaminated noise measurement
            z_contaminated = radar_model.h(truth) + contaminated_gaussian_noise(rng, radar_model.R(), epsilon, kappa)
            z_contaminated_history.append(z_contaminated)

            # To reduce the contamination risk on z_seq which can warp the nees estimates use median of first 5 steps
            # In first five steps, window is short enough that true bearing doesn't approach wrap around so median works
            if t < 5:
                continue
            if t == 5:
                range_est = np.median(np.array(z_contaminated_history[:5])[:, 0])
                bearing_est = np.median(np.array(z_contaminated_history[:5])[:, 1])
                
                px, py = range_est*np.cos(bearing_est), range_est*np.sin(bearing_est)
                x0 = np.array([px, py, 0, 0])
                P0 = np.array(config['initialization']['P0'])

                # Adaptive Position Model is moved to inside to ensure it is fresh at each trial
                rollout_position_model = AdaptivePosition2D(R_init=np.eye(dim_pos)) # Initial R gets overwritten
            
                # Fresh filters for each trial (R is carried, but is overwritten)
                filters = {
                    "B1_denoiser": KalmanFilter(motion=motion_model, measurement=LinearPosition2D(sigma_pos), x0=x0, P0=P0),
                    "B1_B4_rollout": KalmanFilter(motion=motion_model, measurement=rollout_position_model, x0=x0, P0=P0)}

            denoiser_z = None
            rollout_z = None

            # Once enough steps for window, feed current window to b1 and b1_b4 model
            if t >= window: 
                curr_window = np.array(z_contaminated_history[-window:])
                normalized_window = (curr_window - normalization_stats['mean']) / normalization_stats['std']
                # Make tensor with batch dimension
                window_tensor = torch.from_numpy(normalized_window).float().unsqueeze(0)
    
                with torch.no_grad():
                    denoiser_z = model(window_tensor).squeeze(0).numpy()
    
                    # Rollout model position and Cholesky to R
                    rollout_pos, L_rollout = rollout_model.adaptive_denoiser(window_tensor)
                    rollout_z = rollout_pos.squeeze(0).numpy()
                    rollout_position_model.set_R(cov_from_chol(L_rollout).squeeze(0).numpy())

            # Measurements fed to each filter this step (prior to update)
            measurements = {
                "B1_denoiser": denoiser_z,
                "B1_B4_rollout": rollout_z,
            }

            # Predict, update, and record for every filter
            for name, filt in filters.items():
                filt.predict(dt)
                if measurements[name] is not None:
                    filt.update(measurements[name])
                results[name]["pos"].append(filt.x[:2].copy())
                results[name]["nees"].append(nees(filt.x, filt.P, truth))

        # Accumulate sq_err and rmse
        gt = np.array(ground_truth[4:])
        for name in results:
            pos = np.array(results[name]["pos"])
            sq_err = np.sum((pos - gt)**2, axis=1)
            all_sq_err[name].extend(sq_err)
            all_nees[name].extend(results[name]["nees"])

    row_dict = {name: {"rmse": np.sqrt(np.mean(all_sq_err[name])), "mean_nees": np.mean(all_nees[name])} for name in all_sq_err}

    return row_dict


def run_residual_dynamics_rows(config: dict, mc_seeds: list[int]) -> dict[str, dict]:
    """ Creates results for residual dynamics rows
    In: config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    Out: row_dict, {"rmse":..,"mean_nees":..}
    """

    # Load in trained checkpoint
    model, _, checkpoint_config = load_checkpoint(path='data/processed/residual_dynamics_checkpoint.pt',
                                                    build_model_fn=build_residual_dynamics_kf)
    model.eval()

    # Extract checkpoint_config variables
    val_fraction = checkpoint_config['maneuver']['val_fraction']
    seed_rd = checkpoint_config['seed']
    sigma_pos = checkpoint_config['maneuver']['sigma_pos']

     # Load in val_dataset
    _, val_dataset = load_maneuver_dataset(path='data/processed/maneuver_dataset.npz',
                                            K=1,
                                            val_fraction=val_fraction,
                                            seed=seed_rd,
                                            sigma_pos=sigma_pos)
                                        
    trajectories = val_dataset.targets # (n_val, T, 4)

    all_sq_err = []
    all_nees = []

    for seed in mc_seeds:
        rng = np.random.default_rng(seed)  # For reproducibility

        for trajectory in trajectories:

            T = len(trajectory)
            z_seq = trajectory[:, :2] + rng.normal(0, sigma_pos, size=(T, 2))
            x0 = np.concatenate([z_seq[0], np.zeros(2)])
            P0 = np.array(config['initialization']['P0'])

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

            all_sq_err.extend(sq_err_rd)
            all_nees.extend(nees_seq_rd)

    row_dict = {"rmse": np.sqrt(np.mean(all_sq_err)), "mean_nees": np.mean(all_nees)}

    return row_dict

def run_full_pipeline_rows(config: dict, mc_seeds: list[int]) -> dict[str, dict]:
    """ Evaluates FullPipelineModel on its own native stream (sigma_range=100, contaminated noise)
    once on a straight CV trajectory (the "contaminated" column cell) and 
    once on a turning waypoint trajectory matching the maneuvering column's own kinematics (the "maneuvering" column cell)
    In: config, final_comparison.yaml; 
        mc_seeds, config['comparison']['mc_seeds']
    Out: {"contaminated": {"rmse":.., "mean_nees":..}, 
        "maneuvering": {"rmse":.., "mean_nees":..}}
    """

    row_dict = {}

    # Load model, normalization_stats, checkpoint_config
    model, normalization_stats, checkpoint_config = load_checkpoint(path="data/processed/full_pipeline_checkpoint.pt", build_model_fn=build_full_pipeline_model)
    model.eval()

    # Load checkpoint_config variables
    sigma_range = checkpoint_config['scenario']['sigma_range']
    sigma_bearing = checkpoint_config['scenario']['sigma_bearing']
    epsilon = checkpoint_config['scenario']['epsilon']
    kappa = checkpoint_config['scenario']['kappa']
    dt = checkpoint_config['scenario']['dt']
    n_steps = checkpoint_config['scenario']['n_steps']
    min_x = checkpoint_config['scenario']['min_x']
    max_x = checkpoint_config['scenario']['max_x']
    min_y = checkpoint_config['scenario']['min_y']
    max_y = checkpoint_config['scenario']['max_y']

    window = checkpoint_config['dataset']['window']

    turn_prob = checkpoint_config['maneuver']['turn_prob']
    leg_duration_range = checkpoint_config['maneuver']['leg_duration_range']
    omega_range = checkpoint_config['maneuver']['omega_range']

    P0 = np.array(checkpoint_config['initialization']['P0'])

    hidden_dim = checkpoint_config['dynamics_net']['hidden_dim']

    all_sq_err = {"contaminated": [], "maneuvering": []}
    all_nees = {"contaminated": [], "maneuvering": []}

    # Build radar model
    radar_model = RadarRangeBearing(sigma_range=sigma_range, sigma_bearing=sigma_bearing)

    def _run_full_pipeline_on_trajectory(trajectory, model, normalization_stats, radar_model, window, dt, P0, hidden_dim, epsilon, kappa, rng) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper to get sq_err, nees_seq per seed
        """
        dtype = torch.float32

        z_sequence = []
        positions = []

        for i in range(1, n_steps):
            z_sequence.append(radar_model.h(trajectory[i]) + contaminated_gaussian_noise(rng, radar_model.R(), epsilon, kappa))
            positions.append(trajectory[i][:2])
    
        z_sequence = np.array(z_sequence)
        positions = np.array(positions)
    
    
        windows, _ = extract_windows(z_sequence, positions, window)

        zero_velocity = torch.zeros(1, 2) # (1, 2)
        first_meas = windows[0, -1] # (dim_z, )
        px = first_meas[0] * np.cos(first_meas[1])
        py = first_meas[0] * np.sin(first_meas[1])

        px = torch.tensor([[px]], dtype=dtype) # (1, 1)
        py = torch.tensor([[py]], dtype=dtype) # (1, 1)

        x0_init = torch.cat([px, py, zero_velocity], dim=1) # (1, 4)
        P0_init = torch.as_tensor(P0, dtype=dtype).expand(1, -1, -1) # (1, 4, 4)
        h0_init = torch.zeros(1, hidden_dim) # (1, hidden_dim)

        normalized_windows = (windows - normalization_stats['mean']) / normalization_stats['std']

        window_seq = torch.from_numpy(normalized_windows).float().unsqueeze(0) # (1, n_windows, window, dim_z)

        with torch.no_grad():
            x_seq, P_seq, h_seq = model(window_seq, x0_init, P0_init, h0_init)

        x_est = x_seq.squeeze(0).numpy()
        P_est = P_seq.squeeze(0).numpy()

        ground_truth = positions[window-1:]

        sq_err = np.sum((x_est[:, :2] - ground_truth)**2, axis=1)
        
        nees_seq = np.zeros(P_est.shape[0])
        for i in range(P_est.shape[0]):
            error = trajectory[window+i] - x_est[i]
            nees_seq[i] = error @ linalg.solve(P_est[i], error)

        return sq_err, nees_seq

    for seed in mc_seeds:

        rng_c = np.random.default_rng(seed)
        rng_m = np.random.default_rng(seed + 1_000_000)

        x_c = rng_c.uniform(min_x, max_x)
        y_c = rng_c.uniform(min_y, max_y)
        speed_c = rng_c.uniform(5, 30)
        heading_c = rng_c.uniform(0, 2*np.pi)
        x0_true_c = np.array([x_c, y_c, speed_c*np.cos(heading_c), speed_c*np.sin(heading_c)])

        x_m = rng_m.uniform(min_x, max_x)
        y_m = rng_m.uniform(min_y, max_y)
        speed_m = rng_m.uniform(5, 30)
        heading_m = rng_m.uniform(0, 2*np.pi)
        x0_true_m = np.array([x_m, y_m, speed_m*np.cos(heading_m), speed_m*np.sin(heading_m)])
        

        trajectory_c = constant_velocity_trajectory(x0_true_c, dt, n_steps)
        trajectory_m = waypoint_trajectory(x0_true_m, dt, n_steps, rng_m, turn_prob, leg_duration_range, omega_range)

        sq_err_c, nees_c = _run_full_pipeline_on_trajectory(trajectory_c, model, normalization_stats, radar_model, window, dt, P0, hidden_dim, epsilon, kappa, rng_c)
        sq_err_m, nees_m = _run_full_pipeline_on_trajectory(trajectory_m, model, normalization_stats, radar_model, window, dt, P0, hidden_dim, epsilon, kappa, rng_m)

        all_sq_err["contaminated"].extend(sq_err_c)
        all_nees["contaminated"].extend(nees_c)
        all_sq_err["maneuvering"].extend(sq_err_m)
        all_nees["maneuvering"].extend(nees_m)



    row_dict["contaminated"] = {"rmse": np.sqrt(np.mean(all_sq_err["contaminated"])), "mean_nees": np.mean(all_nees["contaminated"])}
    row_dict["maneuvering"] = {"rmse": np.sqrt(np.mean(all_sq_err["maneuvering"])), "mean_nees": np.mean(all_nees["maneuvering"])}

    return row_dict

def run_bearing_only_denoiser_rows(config: dict, mc_seeds: list[int]) -> dict[str, dict]:
    """ Creates results for bearing_only denoiser rows for B1, B1_B4
    In: config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    Out: row_dict, {"B1_denoiser": {"rmse":..,"mean_nees":..}, "B1_B4_rollout": {...}}
    """

    # Load denoiser and adaptive denoiser models, norm stats, and config data from prior created dataset (denoiser_checkpoint)
    model, normalization_stats, checkpoint_config = load_checkpoint(path="data/processed/bearing_denoiser_checkpoint.pt", build_model_fn=build_denoiser)
    model.eval()

    rollout_model, _, _ = load_checkpoint(path="data/processed/bearing_rollout_covariance_checkpoint.pt", build_model_fn=build_rollout_covariance_model)
    rollout_model.eval()

    # Extract checkpoint_config variables
    sigma_bearing = checkpoint_config['scenario_bearing']['sigma_bearing']
    epsilon = checkpoint_config['scenario_bearing']['epsilon']
    kappa = checkpoint_config['scenario_bearing']['kappa']
    window = checkpoint_config['dataset']['window']
    dt = checkpoint_config['scenario_bearing']['dt']
    n_steps = checkpoint_config['scenario_bearing']['n_steps']

    val_fraction = checkpoint_config['dataset']['val_fraction']
    seed = checkpoint_config['seed']
    dim_pos = checkpoint_config['model']['dim_pos']

    batch_size = checkpoint_config['training']['batch_size']

    # Get sigma_pos and default_range_guess from final_comparison.yaml, not checkpoint_config
    sensor_pos = config["scenarios"]["bearing_only"]["sensor_pos"]
    default_range_guess = config["scenarios"]["bearing_only"]["default_range_guess"]

    # Compute sigma_pos empirically
    _, val_dataset, _ = load_denoiser_dataset(path='data/processed/bearing_denoiser_dataset.npz', val_fraction=val_fraction, seed=seed)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # Shuffling unnecessary for val set

    errors = []
    with torch.no_grad():
        for windows, targets in val_loader:
            predictions = model(windows) # Forward pass
            errors.append(predictions-targets) 

    # Error is (batch_size, dim_pos) to become (n_val_samples, dim_pos)
    all_errors = torch.cat(errors, dim=0)

    # This is done as the original RMSE function would inflate bias if present
    std = all_errors.std(dim=0)
    sigma_pos = torch.sqrt(torch.mean(std ** 2)).item()
    
    # Build eoir model
    eoir_model = EOIRBearingOnly(sigma_bearing=sigma_bearing, sensor_pos=sensor_pos)

    # No tuned value so bearing_only EKF will have to do (choosing to focus on other aspects of project)
    motion_model = ConstantVelocity2D(sigma_a=config['tuned']['bearing_only']['EKF'])

    all_sq_err = {"B1_denoiser": [], "B1_B4_rollout": []}
    all_nees = {"B1_denoiser": [], "B1_B4_rollout": []}

    for seed in mc_seeds:
        rng = np.random.default_rng(seed)  # For reproducibility
        
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
        z_contaminated_history = []

        # Set up arrays
        results = {"B1_denoiser": {"pos": [], "nees": []}, 
                   "B1_B4_rollout": {"pos": [], "nees": []}}
        ground_truth = []
        z_contaminated_history = []
        
        # Learned-model outputs (None until the window is full)
        denoiser_z = None
        rollout_z = None

        for t in range(1, n_steps):
            
            truth = trajectory[t]
            ground_truth.append(truth[:2])

            # Contaminated noise measurement
            z_contaminated = eoir_model.h(truth) + contaminated_gaussian_noise(rng, eoir_model.R(), epsilon, kappa)
            z_contaminated_history.append(z_contaminated)

            # To reduce the contamination risk on z_seq which can warp the nees estimates use median of first 5 steps
            # In first five steps, window is short enough that true bearing doesn't approach wrap around so median works
            if t < 5:
                continue
            if t == 5:
                bearing_est = np.median(np.array(z_contaminated_history[:5])[:, 0])  # only one column exists
                
                px = sensor_pos[0] + default_range_guess * np.cos(bearing_est)
                py = sensor_pos[1] + default_range_guess * np.sin(bearing_est)
                
                x0 = np.array([px, py, 0, 0])
                P0 = np.array(config['initialization']['P0_bearing'])

                # Adaptive Position Model is moved to inside to ensure it is fresh at each trial
                rollout_position_model = AdaptivePosition2D(R_init=np.eye(dim_pos)) # Initial R gets overwritten
            
                # Fresh filters for each trial (R is carried, but is overwritten)
                filters = {
                    "B1_denoiser": KalmanFilter(motion=motion_model, measurement=LinearPosition2D(sigma_pos), x0=x0, P0=P0),
                    "B1_B4_rollout": KalmanFilter(motion=motion_model, measurement=rollout_position_model, x0=x0, P0=P0)}

            denoiser_z = None
            rollout_z = None

            # Once enough steps for window, feed current window to b1 and b1_b4 model
            if t >= window: 
                curr_window = np.array(z_contaminated_history[-window:])
                normalized_window = (curr_window - normalization_stats['mean']) / normalization_stats['std']
                # Make tensor with batch dimension
                window_tensor = torch.from_numpy(normalized_window).float().unsqueeze(0)
    
                with torch.no_grad():
                    denoiser_z = model(window_tensor).squeeze(0).numpy()
    
                    # Rollout model position and Cholesky to R
                    rollout_pos, L_rollout = rollout_model.adaptive_denoiser(window_tensor)
                    rollout_z = rollout_pos.squeeze(0).numpy()
                    rollout_position_model.set_R(cov_from_chol(L_rollout).squeeze(0).numpy())

            # Measurements fed to each filter this step (prior to update)
            measurements = {
                "B1_denoiser": denoiser_z,
                "B1_B4_rollout": rollout_z,
            }

            # Predict, update, and record for every filter
            for name, filt in filters.items():
                filt.predict(dt)
                if measurements[name] is not None:
                    filt.update(measurements[name])
                results[name]["pos"].append(filt.x[:2].copy())
                results[name]["nees"].append(nees(filt.x, filt.P, truth))

        # Accumulate sq_err and rmse
        gt = np.array(ground_truth[4:])
        for name in results:
            pos = np.array(results[name]["pos"])
            sq_err = np.sum((pos - gt)**2, axis=1)
            all_sq_err[name].extend(sq_err)
            all_nees[name].extend(results[name]["nees"])

    row_dict = {name: {"rmse": np.sqrt(np.mean(all_sq_err[name])), "mean_nees": np.mean(all_nees[name])} for name in all_sq_err}

    return row_dict

def run_full_pipeline_bearing_rows(config: dict, mc_seeds: list[int]) -> dict[str, dict]:
    """ Evaluates FullPipelineModel on its bearing-only native stream (EOIRBearingOnly, contaminated noise)
    In: config, final_comparison.yaml; mc_seeds, config['comparison']['mc_seeds']
    Out: {"rmse":.., "mean_nees":..}
    """

    # Load model, normalization_stats, checkpoint_config
    model, normalization_stats, checkpoint_config = load_checkpoint(path="data/processed/full_pipeline_bearing_checkpoint.pt", build_model_fn=build_full_pipeline_bearing_model)
    model.eval()

    # Load checkpoint_config variables
    sigma_bearing = checkpoint_config['scenario_bearing']['sigma_bearing']
    epsilon = checkpoint_config['scenario_bearing']['epsilon']
    kappa = checkpoint_config['scenario_bearing']['kappa']
    dt = checkpoint_config['scenario_bearing']['dt']
    n_steps = checkpoint_config['scenario_bearing']['n_steps']
    min_x = checkpoint_config['scenario_bearing']['min_x']
    max_x = checkpoint_config['scenario_bearing']['max_x']
    min_y = checkpoint_config['scenario_bearing']['min_y']
    max_y = checkpoint_config['scenario_bearing']['max_y']

    window = checkpoint_config['dataset']['window']

    P0 = np.array(checkpoint_config['initialization']['P0_bearing'])

    hidden_dim = checkpoint_config['dynamics_net']['hidden_dim']

    # Get sigma_pos and default_range_guess from final_comparison.yaml, not checkpoint_config
    sensor_pos = config["scenarios"]["bearing_only"]["sensor_pos"]
    default_range_guess = config["scenarios"]["bearing_only"]["default_range_guess"]

    all_sq_err = []
    all_nees = []

    # Build eoir model
    eoir_model = EOIRBearingOnly(sigma_bearing=sigma_bearing, sensor_pos=sensor_pos)

    def _run_full_pipeline_on_trajectory(trajectory, model, normalization_stats, eoir_model, window, dt, P0, hidden_dim, epsilon, kappa, rng) -> tuple[np.ndarray, np.ndarray]:
        """
        Helper to get sq_err, nees_seq per seed
        """
        dtype = torch.float32

        z_sequence = []
        positions = []

        for i in range(1, n_steps):
            z_sequence.append(eoir_model.h(trajectory[i]) + contaminated_gaussian_noise(rng, eoir_model.R(), epsilon, kappa))
            positions.append(trajectory[i][:2])
    
        z_sequence = np.array(z_sequence)
        positions = np.array(positions)
    
        windows, _ = extract_windows(z_sequence, positions, window)

        zero_velocity = torch.zeros(1, 2) # (1, 2)
        first_meas = windows[0, -1] # (dim_z, )
        bearing_est = first_meas[0]
        px = sensor_pos[0] + default_range_guess * np.cos(bearing_est)
        py = sensor_pos[1] + default_range_guess * np.sin(bearing_est)

        px = torch.tensor([[px]], dtype=dtype) # (1, 1)
        py = torch.tensor([[py]], dtype=dtype) # (1, 1)

        x0_init = torch.cat([px, py, zero_velocity], dim=1) # (1, 4)
        P0_init = torch.as_tensor(P0, dtype=dtype).expand(1, -1, -1) # (1, 4, 4)
        h0_init = torch.zeros(1, hidden_dim) # (1, hidden_dim)

        normalized_windows = (windows - normalization_stats['mean']) / normalization_stats['std']

        window_seq = torch.from_numpy(normalized_windows).float().unsqueeze(0) # (1, n_windows, window, dim_z)

        with torch.no_grad():
            x_seq, P_seq, h_seq = model(window_seq, x0_init, P0_init, h0_init)

        x_est = x_seq.squeeze(0).numpy()
        P_est = P_seq.squeeze(0).numpy()

        ground_truth = positions[window-1:]

        sq_err = np.sum((x_est[:, :2] - ground_truth)**2, axis=1)
        
        nees_seq = np.zeros(P_est.shape[0])
        for i in range(P_est.shape[0]):
            error = trajectory[window+i] - x_est[i]
            nees_seq[i] = error @ linalg.solve(P_est[i], error)

        return sq_err, nees_seq

    for seed in mc_seeds:

        rng = np.random.default_rng(seed)

        x = rng.uniform(min_x, max_x)
        y = rng.uniform(min_y, max_y)
        speed = rng.uniform(5, 30)
        heading = rng.uniform(0, 2*np.pi)
        x0_true = np.array([x, y, speed*np.cos(heading), speed*np.sin(heading)])


        trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

        sq_err, nees = _run_full_pipeline_on_trajectory(trajectory, model, normalization_stats, eoir_model, window, dt, P0, hidden_dim, epsilon, kappa, rng)

        all_sq_err.extend(sq_err)
        all_nees.extend(nees)


    row_dict = {"rmse": np.sqrt(np.mean(all_sq_err)), "mean_nees": np.mean(all_nees)}

    return row_dict


# Table functions below

def pivot_results(results: dict) -> dict[str, dict[str, dict | None]]:
    """ Builds methods by scenarios grid
    In:  results, {"benign": {...}, "maneuvering": {...}, "contaminated": {...},
                 "bearing_only": {...}, "dual_sensor_fusion": {...}}
    Out: table[row_label][scenario_name] -> stat_dict | None
        row_label is the union of every method name seen across all five scenarios
        A cell is None wherever that row's method doesn't apply to that column's scenario
    """

    # Hardcoded row order
    row_order = ["EKF", 
                "UKF", 
                "IMM", 
                "B1_denoiser", 
                "B1_B4_rollout", 
                "B2_residual_dynamics", 
                "full_pipeline",
                "sequential_EKF", 
                "sequential_UKF",
                "naive_EKF", 
                "naive_UKF", 
                "ci_EKF", 
                "ci_UKF"]

    table = {}

    # table is organized as table[row_label][scenario_name] = stat_dict
    for row in row_order:
        table[row] = {}
        for scenario, scenario_dict in results.items():
            table[row][scenario] = scenario_dict.get(row)  # None if this row doesn't apply here
    
    return table

def print_table(table: dict[str, dict[str, dict | None]]) -> None:
    """ Prints table cleanly in console
    In:  table, output of pivot_results;
    out: None (prints to console)
    """

    # Ensure order
    scenario_order = [
        "benign",
        "maneuvering",
        "contaminated",
        "bearing_only",
        "dual_sensor_fusion",
        "heavy_clutter"]

    # Column formatting
    col_width = 25

    # Print label
    print("Cells show RMSE / Mean NEES")
    print("heavy_clutter shows Mean # of ID Switches / Mean False Track Rate")
    print("")

    # Print header
    header = f"{'Method':<{col_width}}"
    for scenario in scenario_order:
        header += f"{scenario:>{col_width}}"

    print(header)
    print("-" * len(header))

    # Cycle through each method
    for row_label, scenario_dict in table.items():

        row = f"{row_label:<{col_width}}"

        # Store maneuvering regime statistics for separate printing
        maneuvering_notes = None

        for scenario in scenario_order:
            cell = scenario_dict.get(scenario)

            if cell is None:
                cell_str = "N/A"

            # Heavy clutter uses id_switches and false_track_rate
            elif scenario == "heavy_clutter":
                cell_str = (
                    f"{cell['id_switches']:.3f} / "
                    f"{cell['false_track_rate']:.3f}"
                )

            else:
                # Maneuvering contains nested regime statistics
                if scenario == "maneuvering" and "overall" in cell:
                    overall = cell["overall"]
                    maneuvering_notes = cell

                else:
                    overall = cell

                cell_str = (
                    f"{overall['rmse']:.3f} / "
                    f"{overall['mean_nees']:.3f}"
                )

            row += f"{cell_str:>{col_width}}"

        print(row)

        # Print maneuvering regime statistics underneath the row
        if maneuvering_notes is not None:
            straight = maneuvering_notes["straight"]
            turning = maneuvering_notes["turning"]

            print(
                f"{'':<{2*col_width}}"
                f"  Straight: {straight['rmse']:.3f} / "
                f"{straight['mean_nees']:.3f}"
            )

            print(
                f"{'':<{2*col_width}}"
                f"  Turning: {turning['rmse']:.3f} / "
                f"{turning['mean_nees']:.3f}"
            )

        print("")

def save_table(table: dict[str, dict[str, dict | None]], path: Path) -> None:
    """ Saves
    In: table, output of pivot_results;
        path, path for file
    Out: None (writes path)
    """

    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)

    # Write table to JSON
    with open(path, "w") as f:
        json.dump(table, f, indent=2)

    print(f"Comparison table saved to: {path}")


# Main function
def main() -> None:
    """
    Calls all 6 row functions and assembles results into one dict
    Prints and saves comparison table (using pivot_results)
    """

    mc_seeds = config["comparison"]["mc_seeds"]

    # Results for each row
    print("Running full pipeline rows")
    full_pipeline_results = run_full_pipeline_rows(config, mc_seeds)
    print("Running full pipeline bearing only rows")
    full_pipeline_bearing_results = run_full_pipeline_bearing_rows(config, mc_seeds)
    print("Running benign rows")
    benign_results = run_benign_rows(config, mc_seeds)
    print("Running maneuvering rows")
    maneuvering_results = run_maneuvering_rows(config, mc_seeds)
    print("Run residual dynamics rows")
    residual_dynamics_results = run_residual_dynamics_rows(config, mc_seeds)
    print("Running contaminated rows")
    contaminated_results = run_contaminated_rows(config, mc_seeds)
    print("Running denoiser rows")
    denoiser_results = run_denoiser_rows(config, mc_seeds)
    print("Running bearing only rows")
    bearing_only_results = run_bearing_only_rows(config, mc_seeds)
    print("Running bearing only denoiser rows")
    bearing_only_denoiser_results = run_bearing_only_denoiser_rows(config, mc_seeds)
    print("Running dual sensor fusion rows")
    dual_sensor_fusion_results = run_dual_sensor_fusion_rows(config, mc_seeds)
    print("Running heavy clutter rows")
    heavy_clutter_results = run_heavy_clutter_rows(config, mc_seeds)

    # Accumulate into results dict for pivot_results table
    results = {}

    results["benign"] = benign_results

    results["maneuvering"] = maneuvering_results
    results["maneuvering"]["B2_residual_dynamics"] = residual_dynamics_results
    results["maneuvering"]["full_pipeline"] = full_pipeline_results["maneuvering"]

    results["contaminated"] = contaminated_results
    results["contaminated"].update(denoiser_results)
    results["contaminated"]["full_pipeline"] = full_pipeline_results["contaminated"]

    results["bearing_only"] = bearing_only_results
    results["bearing_only"].update(bearing_only_denoiser_results)
    results["bearing_only"]["full_pipeline"] = full_pipeline_bearing_results
    # Residual dynamics correction never touches measurement dimensionality so same results
    results["bearing_only"]["B2_residual_dynamics"] = residual_dynamics_results

    results["dual_sensor_fusion"] = dual_sensor_fusion_results

    results["heavy_clutter"] = heavy_clutter_results

    table = pivot_results(results)

    save_table(table, Path("data/processed/final_comparison_table.json"))
    print_table(table)

if __name__ == "__main__": main()


