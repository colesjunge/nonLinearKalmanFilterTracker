from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "final_comparison.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import numpy as np
from typing import Callable

from tracking.filters.base import BayesFilter
from fusion.eval.scenarios import build_maneuvering_scenario, build_dual_sensor_fusion_scenario
from fusion.eval.filters import build_dual_sensor_sequential_filter, build_dual_sensor_track_pair
from fusion.eval.rollout import run_filter_on_trajectory, run_sequential_fusion_on_trajectory, run_track_level_fusion_on_trajectory
from fusion.eval.metrics import classify_maneuver_segments
from fusion.models.radar import RadarRangeBearing
from fusion.models.eoir import EOIRBearingOnly


def mean_nees_for_cell(sigma_a: float, scenario_fn: Callable, filter_builder_fn: Callable, config: dict, tuning_seeds: list[int], mask_fn: Callable | None = None) -> float:
    """ Creates respective filter and runs trajectories to determine mean nees
    In: sigma_a, the value being swept/root-found this call;
        scenario_fn, one of the functions from fusion.eval.scenarios;
        filter_builder_fn(sigma_a, trajectory, z_seq), has no scenario specific knowledge;
        config, final_comparison.yaml;
        tuning_seeds, config['tuning_seeds'];
        mask_fn(trajectory), bool mask (T,) | None, Only used for maneuvering's splits
    Out: mean_nees, mean NEES pooled across every tuning seed
    """

    all_nees = []
    mask = None

    for seed in tuning_seeds:

        # Set rng
        rng = np.random.default_rng(seed)

        # Build relevant trajectory and z_seq for scenario (dual_sensor_fusion seperate)
        trajectory, z_seq = scenario_fn(rng, config)

        # Create mask if relevant
        if mask_fn:
            mask = mask_fn(trajectory)
        
        # Create filter
        filt = filter_builder_fn(sigma_a, trajectory, z_seq)

        # Calculate nees_seq for this seed
        _, nees_seq = run_filter_on_trajectory(filt, trajectory, z_seq, config['dt'])

        # Apply mask if relevant
        if mask is not None:
            nees_seq = nees_seq[mask]

        # Accumulate nees
        all_nees.extend(nees_seq)

    mean_nees = np.mean(all_nees)

    return mean_nees

def compute_omega(config: dict, tuning_seeds: list[int]) -> float:
    """ Pools turning angular rates across seeds to determine optimal omega (median magnitude)
    In: config, final_comparison.yaml;
        tuning_seeds, config['tuning_seeds'];
    Out: omega, median|angular_rates|
    """

    all_angular_rates = []

    for seed in tuning_seeds:

        # Set rng
        rng = np.random.default_rng(seed)

        # Build relevant trajectory for maneuvering scenario (dual_sensor_fusion seperate)
        trajectory, _ = build_maneuvering_scenario(rng, config)

        # Create is_turning and angular_rates
        is_turning, angular_rates = classify_maneuver_segments(trajectory, config['dt'])

        # Apply is_turning mask
        turning_angular_rates = angular_rates[is_turning]

        # Accumulate turning angular rates
        all_angular_rates.extend(turning_angular_rates)

    omega = np.median(np.abs(all_angular_rates))

    return omega

def straight_mask(trajectory: np.ndarray) -> np.ndarray:
    """
    Create mask for straight legs
    """

    is_turning, _ = classify_maneuver_segments(trajectory, config['dt'])

    return ~is_turning

def ct_plus_mask(trajectory: np.ndarray) -> np.ndarray:
    """
    Create mask for CCW turning legs
    """

    is_turning, angular_rates = classify_maneuver_segments(trajectory, config['dt'])

    return is_turning & (angular_rates > 0)

def ct_minus_mask(trajectory: np.ndarray) -> np.ndarray:
    """
    Create mask for CW turning legs
    """

    is_turning, angular_rates = classify_maneuver_segments(trajectory, config['dt'])

    return is_turning & (angular_rates < 0)

def mean_nees_for_sequential(sigma_a: float, filter_cls: type[BayesFilter], config: dict, tuning_seeds: list[int]) -> float:
    """ Creates filters and sequential fusion and runs trajectories to determine mean nees
    In: sigma_a, the value being swept/root-found this call;
        filter_cls, filter builder (UKF or EKF);
        config, final_comparison.yaml;
        tuning_seeds, config['tuning_seeds'];
    Out: mean_nees, mean NEES pooled across every tuning seed
    """

    all_nees = []

    # Get appropriate parameters
    sigma_range = config["scenarios"]["dual_sensor_fusion"]["sigma_range"]
    sigma_bearing = config["scenarios"]["dual_sensor_fusion"]["sigma_bearing"]
    radar_pos = config["scenarios"]["dual_sensor_fusion"]["radar_pos"]
    eoir_pos = config["scenarios"]["dual_sensor_fusion"]["eoir_pos"]

    for seed in tuning_seeds:

        # Set rng
        rng = np.random.default_rng(seed)

        # Build relevant trajectory and z_seqs for dual_sensor_sequential
        trajectory, z_radar_seq, z_eoir_seq = build_dual_sensor_fusion_scenario(rng, config)

        # Create filter and models
        filt = build_dual_sensor_sequential_filter(sigma_a, trajectory, z_radar_seq, z_eoir_seq, filter_cls, config)
        radar_model = RadarRangeBearing(sigma_range, sigma_bearing, sensor_pos=np.array(radar_pos))
        eoir_model = EOIRBearingOnly(sigma_bearing, sensor_pos=np.array(eoir_pos))
        
        # Calculate nees_seq for this seed
        _, nees_seq = run_sequential_fusion_on_trajectory(filt, trajectory, z_radar_seq, z_eoir_seq, config['dt'], radar_model, eoir_model)

        # Accumulate nees
        all_nees.extend(nees_seq)

    mean_nees = np.mean(all_nees)

    return mean_nees

def mean_nees_for_track_level(sigma_a: float, filter_cls: type[BayesFilter], fuse_fn: Callable, config: dict, tuning_seeds: list[int]) -> float:
    """ Creates filters and track level fusion and runs trajectories to determine mean nees
    In: sigma_a, the value being swept/root-found this call;
        filter_cls, filter builder (UKF or EKF);
        fuse_fn(x1, P1, x2, P2) -> (x_fused, P_fused), either fuse_naive or covariance_intersection;
        config, final_comparison.yaml;
        tuning_seeds, config['tuning_seeds'];
    Out: mean_nees, mean NEES pooled across every tuning seed
    """

    all_nees = []

    for seed in tuning_seeds:

        # Set rng
        rng = np.random.default_rng(seed)

        # Build relevant trajectory and z_seqs for dual_sensor_track_level
        trajectory, z_radar_seq, z_eoir_seq = build_dual_sensor_fusion_scenario(rng, config)

        # Create filters
        filt_radar, filt_eoir = build_dual_sensor_track_pair(sigma_a, trajectory, z_radar_seq, z_eoir_seq, filter_cls, config)
        
        # Calculate nees_seq for this seed
        _, nees_seq = run_track_level_fusion_on_trajectory(filt_radar, filt_eoir, trajectory, z_radar_seq, z_eoir_seq, config['dt'], fuse_fn)
       
        # Accumulate nees
        all_nees.extend(nees_seq)

    mean_nees = np.mean(all_nees)

    return mean_nees