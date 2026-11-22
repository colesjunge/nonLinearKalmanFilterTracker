from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "final_comparison.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
from typing import Callable

from tracking.filters.base import BayesFilter
from fusion.eval.metrics import classify_maneuver_segments
from fusion.eval.scenarios import build_dual_sensor_fusion_scenario
from fusion.eval.filters import build_dual_sensor_sequential_filter, build_dual_sensor_track_pair
from fusion.eval.rollout import run_filter_on_trajectory, run_sequential_fusion_on_trajectory, run_track_level_fusion_on_trajectory
from fusion.models.radar import RadarRangeBearing
from fusion.models.eoir import EOIRBearingOnly
from fusion.filters.ukf import UnscentedKalmanFilter
from fusion.filters.ekf import ExtendedKalmanFilter


# Helper to calculate statistics
def calculate_stats(nees, sq_err):
    return {"rmse": np.sqrt(np.mean(sq_err)), "mean_nees": np.mean(nees)}

def run_comparison_for_cell(scenario_fn: Callable, filter_builder_fn: Callable, config: dict, mc_seeds: list[int], split_by_regime: bool = False) -> dict:
    """ Creates respective filter and runs trajectories to determine mean nees and rmse
    In: scenario_fn, one of the functions from fusion.eval.scenarios;
        filter_builder_fn(trajectory, z_seq), has scenario specific knowledge (sigma_a);
        config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
        split_by_regime, True only for maneuvering's EKF/UKF/IMM rows
    Out: stat_dict {"rmse": rmse, "mean_nees": mean_nees}, mean NEES and RMSE pooled across every mc trial
                    {"straight": {...}, "turning": {...}, "overall": {...}} when split_by_regime=True
    """

    all_nees = []
    all_sq_err = []

    stat_dict ={}

    regime_data = {
        "straight": {"nees": [], "sq_err": []},
        "turning": {"nees": [], "sq_err": []}
    }

    for seed in mc_seeds:

        rng = np.random.default_rng(seed)

        # Generate trajectory and measurements
        trajectory, z_seq = scenario_fn(rng, config)

        # Build filter
        filt = filter_builder_fn(trajectory, z_seq)

        # Run filter
        x_est_seq, nees_seq = run_filter_on_trajectory(filt, trajectory, z_seq, config["dt"])

        # Calculate squared errors
        sq_err = np.sum((x_est_seq[:, :2] - trajectory[:, :2]) ** 2, axis=1)  # (T,)

        if split_by_regime:
            is_turning, _ = classify_maneuver_segments(trajectory, config["dt"])
            regime_data["straight"]["nees"].extend(nees_seq[~is_turning])
            regime_data["straight"]["sq_err"].extend(sq_err[~is_turning])
            regime_data["turning"]["nees"].extend(nees_seq[is_turning])
            regime_data["turning"]["sq_err"].extend(sq_err[is_turning])
        
        all_nees.extend(nees_seq)
        all_sq_err.extend(sq_err)

    if split_by_regime:
        for regime in regime_data:
            stat_dict[regime] = calculate_stats(regime_data[regime]["nees"], regime_data[regime]["sq_err"])        
        stat_dict["overall"] = calculate_stats(all_nees, all_sq_err)
    else:
        stat_dict = calculate_stats(all_nees, all_sq_err)

    return stat_dict

def run_comparison_for_sequential(filter_cls: type[BayesFilter], config: dict, mc_seeds: list[int]) -> dict:
    """ Creates filters and sequential fusion and runs trajectories to determine mean nees and rmse
    In: filter_cls, filter builder (UKF or EKF);
        config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    Out: stat_dict {"rmse": rmse, "mean_nees": mean_nees}, mean NEES and RMSE pooled across every mc trial
    """

    all_nees = []
    all_sq_err = []

    stat_dict ={}

    # Get appropriate parameters
    sigma_range = config["scenarios"]["dual_sensor_fusion"]["sigma_range"]
    sigma_bearing = config["scenarios"]["dual_sensor_fusion"]["sigma_bearing"]
    radar_pos = config["scenarios"]["dual_sensor_fusion"]["radar_pos"]
    eoir_pos = config["scenarios"]["dual_sensor_fusion"]["eoir_pos"]

    if filter_cls == ExtendedKalmanFilter:
        sigma_a = config["tuned"]["dual_sensor_fusion"]["sequential"]["EKF"]
    elif filter_cls == UnscentedKalmanFilter:
        sigma_a = config["tuned"]["dual_sensor_fusion"]["sequential"]["UKF"]

    for seed in mc_seeds:

        # Set rng
        rng = np.random.default_rng(seed)

        # Build relevant trajectory and z_seqs for dual_sensor_sequential
        trajectory, z_radar_seq, z_eoir_seq = build_dual_sensor_fusion_scenario(rng, config)

        # Create filter and models
        filt = build_dual_sensor_sequential_filter(sigma_a, trajectory, z_radar_seq, z_eoir_seq, filter_cls, config)
        radar_model = RadarRangeBearing(sigma_range, sigma_bearing, sensor_pos=np.array(radar_pos))
        eoir_model = EOIRBearingOnly(sigma_bearing, sensor_pos=np.array(eoir_pos))
        
        # Calculate nees_seq for this seed
        x_est_seq, nees_seq = run_sequential_fusion_on_trajectory(filt, trajectory, z_radar_seq, z_eoir_seq, config['dt'], radar_model, eoir_model)

        # Calculate squared errors
        sq_err = np.sum((x_est_seq[:, :2] - trajectory[:, :2]) ** 2, axis=1)  # (T,)
        
        # Accumulate nees
        all_nees.extend(nees_seq)
        all_sq_err.extend(sq_err)

    stat_dict = calculate_stats(all_nees, all_sq_err)

    return stat_dict


def run_comparison_for_track_level(filter_cls: type[BayesFilter], fuse_fn: Callable, config: dict, mc_seeds: list[int]) -> dict:
    """ Creates filters and sequential fusion and runs trajectories to determine mean nees and rmse
     In: filter_cls, filter builder (UKF or EKF);
        fuse_fn(x1, P1, x2, P2) -> (x_fused, P_fused), either fuse_naive or covariance_intersection;
        config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    Out: stat_dict {"rmse": rmse, "mean_nees": mean_nees}, mean NEES and RMSE pooled across every mc trial
    """

    all_nees = []
    all_sq_err = []

    stat_dict ={}

    # Get appropriate parameters
    if filter_cls == ExtendedKalmanFilter:
        sigma_a = config["tuned"]["dual_sensor_fusion"]["track_level"]["EKF"]
    elif filter_cls == UnscentedKalmanFilter:
        sigma_a = config["tuned"]["dual_sensor_fusion"]["track_level"]["UKF"]

    for seed in mc_seeds:

        # Set rng
        rng = np.random.default_rng(seed)

        # Build relevant trajectory and z_seqs for dual_sensor_track_level
        trajectory, z_radar_seq, z_eoir_seq = build_dual_sensor_fusion_scenario(rng, config)

        # Create filters
        filt_radar, filt_eoir = build_dual_sensor_track_pair(sigma_a, trajectory, z_radar_seq, z_eoir_seq, filter_cls, config)
        
        # Calculate nees_seq for this seed
        x_est_seq, nees_seq = run_track_level_fusion_on_trajectory(filt_radar, filt_eoir, trajectory, z_radar_seq, z_eoir_seq, config['dt'], fuse_fn)

        # Calculate squared errors
        sq_err = np.sum((x_est_seq[:, :2] - trajectory[:, :2]) ** 2, axis=1)  # (T,)
        
        # Accumulate nees
        all_nees.extend(nees_seq)
        all_sq_err.extend(sq_err)

    stat_dict = calculate_stats(all_nees, all_sq_err)

    return stat_dict