from pathlib import Path
import yaml
config_path = Path(__file__).resolve().parent.parent / "configs" / "final_comparison.yaml"
with open(config_path) as f:
    config = yaml.safe_load(f)

import json
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

from fusion.eval.metrics import classify_maneuver_segments
from fusion.eval.rollout import run_filter_on_trajectory

from fusion.eval.scenarios import build_benign_scenario, build_maneuvering_scenario, build_contaminated_scenario, build_bearing_only_scenario
from fusion.eval.filters import build_benign_filter, build_maneuvering_filter, build_contaminated_filter, build_imm_filter, build_bearing_only_filter
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.ukf import UnscentedKalmanFilter

from fusion.eval.scenarios import build_dual_sensor_fusion_scenario
from fusion.eval.filters import build_dual_sensor_sequential_filter, build_dual_sensor_track_pair
from fusion.eval.rollout import run_filter_on_trajectory, run_sequential_fusion_on_trajectory, run_track_level_fusion_on_trajectory
from fusion.models.radar import RadarRangeBearing
from fusion.models.eoir import EOIRBearingOnly
from fusion.fuse.naive import fuse_naive
from fusion.fuse.covariance_intersection import covariance_intersection

def compute_per_seed_diagnostics(scenario_fn: Callable, filter_builder_fn: Callable, config: dict, mc_seeds: list[int], split_by_regime: bool = False, sensor_pos: np.ndarray = np.zeros(2)) -> dict:
    """
    Compute mean_nees and min_range for each seed (not pooled)
    To help diagnose how close to spawn tracks affect NEES (not applicable to dual sensor fusion)
    In: scenario_fn, one of the functions from fusion.eval.scenarios;
        filter_builder_fn(trajectory, z_seq), has scenario specific knowledge (sigma_a);
        config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
        split_by_regime, True only for maneuvering's EKF/UKF/IMM rows
        sensor_pos, [0,0] for all except bearing_only
    Out: stat_dict {"seeds": np.ndarray (N,), "mean_nees": np.ndarray (N,), "min_range": np.ndarray (N,)}
        stat_dict {"seeds": np.ndarray (N,), "mean_nees": np.ndarray (N,), "min_range": np.ndarray (N,)}, mean NEES and min_range for each mc seed
                    {"straight": {...}, "turning": {...}, "overall": {...}} when split_by_regime=True
    
    """

    all_nees = []
    all_min_range = []

    stat_dict = {}

    regime_data = {"straight": {"nees": [], "min_range": []},
                    "turning": {"nees": [], "min_range": []}}

    for seed in mc_seeds:

        rng = np.random.default_rng(seed)

        # Generate trajectory and measurements
        trajectory, z_seq = scenario_fn(rng, config)

        # Build filter
        filt = filter_builder_fn(trajectory, z_seq)

        # Run filter
        _, nees_seq = run_filter_on_trajectory(filt, trajectory, z_seq, config["dt"])
        
        # Compute ranges
        ranges = np.hypot(trajectory[:,0] - sensor_pos[0], trajectory[:,1] - sensor_pos[1])

        # Append mean_nees and min_range
        all_nees.append(nees_seq.mean())
        all_min_range.append(ranges.min())

        # Protect against empty masks
        if split_by_regime:
            is_turning, _ = classify_maneuver_segments(trajectory, config["dt"])
            
            if (~is_turning).any():
                regime_data["straight"]["nees"].append(nees_seq[~is_turning].mean())
                regime_data["straight"]["min_range"].append(ranges[~is_turning].min())
            else:
                regime_data["straight"]["nees"].append(np.nan)
                regime_data["straight"]["min_range"].append(np.nan)

            if is_turning.any():
                regime_data["turning"]["nees"].append(nees_seq[is_turning].mean())
                regime_data["turning"]["min_range"].append(ranges[is_turning].min())
            else:
                regime_data["turning"]["nees"].append(np.nan)
                regime_data["turning"]["min_range"].append(np.nan)


    if split_by_regime:
        stat_dict["straight"] = {"seeds": np.array(mc_seeds), 
                                 "mean_nees": np.array(regime_data["straight"]["nees"]), 
                                 "min_range": np.array(regime_data["straight"]["min_range"])}
        stat_dict["turning"] = {"seeds": np.array(mc_seeds), 
                                 "mean_nees": np.array(regime_data["turning"]["nees"]), 
                                 "min_range": np.array(regime_data["turning"]["min_range"])}
        stat_dict["overall"] = {"seeds": np.array(mc_seeds), 
                                 "mean_nees": np.array(all_nees), 
                                 "min_range": np.array(all_min_range)}
    else:
        stat_dict = {"seeds": np.array(mc_seeds), 
                     "mean_nees": np.array(all_nees), 
                     "min_range": np.array(all_min_range)}

    return stat_dict

def compute_per_seed_diagnostics_dual_sensor(filter_cls, fuse_style: str, config: dict, mc_seeds: list[int]) -> dict:
    """
    Compute mean_nees and min_range for each seed (not pooled)
    To help diagnose how close to spawn tracks affect NEES (not applicable to dual sensor fusion)
    In: filter_cls, filter builder (UKF or EKF);
        fuse_style, "sequential", "naive", or "ci";
        config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    Out: sstat_dict {"seeds": np.ndarray (N,), "mean_nees": np.ndarray (N,), "min_range": np.ndarray (N,)}
    """

    all_nees = []
    all_min_range = []
    

    stat_dict ={}

    # Get appropriate parameters
    if filter_cls == ExtendedKalmanFilter:
        if fuse_style != "sequential":
            sigma_a = config["tuned"]["dual_sensor_fusion"]["track_level"]["EKF"]
        else:
            sigma_a = config["tuned"]["dual_sensor_fusion"]["sequential"]["EKF"]

    elif filter_cls == UnscentedKalmanFilter:
        if fuse_style != "sequential":
            sigma_a = config["tuned"]["dual_sensor_fusion"]["track_level"]["UKF"]
        else:
            sigma_a = config["tuned"]["dual_sensor_fusion"]["sequential"]["UKF"]

    sigma_range = config["scenarios"]["dual_sensor_fusion"]["sigma_range"]
    sigma_bearing = config["scenarios"]["dual_sensor_fusion"]["sigma_bearing"]
    radar_pos = config["scenarios"]["dual_sensor_fusion"]["radar_pos"]
    eoir_pos = config["scenarios"]["dual_sensor_fusion"]["eoir_pos"]

    for seed in mc_seeds:
        
        # Set rng
        rng = np.random.default_rng(seed)

         # Build relevant trajectory and z_seqs for dual_sensor_track_level
        trajectory, z_radar_seq, z_eoir_seq = build_dual_sensor_fusion_scenario(rng, config)
        

        if fuse_style == "sequential":
            # Create filter and models
            filt = build_dual_sensor_sequential_filter(sigma_a, trajectory, z_radar_seq, z_eoir_seq, filter_cls, config)
            radar_model = RadarRangeBearing(sigma_range, sigma_bearing, sensor_pos=np.array(radar_pos))
            eoir_model = EOIRBearingOnly(sigma_bearing, sensor_pos=np.array(eoir_pos))
                    
            # Calculate nees_seq for this seed
            _, nees_seq = run_sequential_fusion_on_trajectory(filt, trajectory, z_radar_seq, z_eoir_seq, config['dt'], radar_model, eoir_model)
    
        else:
            # Get correct fuse_fn
            if fuse_style == "naive":
                fuse_fn = fuse_naive
            elif fuse_style == "ci":
                fuse_fn = covariance_intersection
            # Create filter and models
            filt_radar, filt_eoir = build_dual_sensor_track_pair(sigma_a, trajectory, z_radar_seq, z_eoir_seq, filter_cls, config)
            
            # Calculate nees_seq for this seed
            _, nees_seq = run_track_level_fusion_on_trajectory(filt_radar, filt_eoir, trajectory, z_radar_seq, z_eoir_seq, config['dt'], fuse_fn)

        # Compute ranges (distance to closest sensoer)
        ranges = np.minimum(
            np.hypot(trajectory[:,0]-radar_pos[0], trajectory[:,1]-radar_pos[1]),
            np.hypot(trajectory[:,0]-eoir_pos[0], trajectory[:,1]-eoir_pos[1])
            )

        # Append mean_nees and min_range
        all_nees.append(nees_seq.mean())
        all_min_range.append(ranges.min())

    stat_dict = {"seeds": np.array(mc_seeds), 
                "mean_nees": np.array(all_nees), 
                "min_range": np.array(all_min_range)}
    
    return stat_dict



def summarize_outliers(diagnostics: dict, min_range_threshold: float, nees_threshold: float = 20.0) -> dict:
    """ Classifies each seed against both thresholds
    min_range < min_range_threshold is labeled close to spawn
    mean_nees > nees_threshold is labeled catastrophic
    Computes NEES for all and without catastrophic as well as median

    in: diagnostics, output of compute_per_seed_diagnostics;
        min_range_threshold, config['scenarios']['...']['sigma_range']
        nees_threshold, per-seed mean NEES above this is flagged "catastrophic" (20 for now)
    out: outlier_dict {"n_total":.., "n_close_spawn":.., "frac_close_spawn":.., "n_catastrophic":..,
        "frac_catastrophic":.., "pooled_nees_all":.., "pooled_nees_excl_catastrophic":..,
        "median_per_seed_nees":.., "outlier_seeds": [seed, ...]}

    """
    # Total count of non_nan entries
    n_total = int(np.sum(~np.isnan(diagnostics["mean_nees"])))

    # Masks
    close_spawn_mask = diagnostics["min_range"] < min_range_threshold
    catastrophic_mask = diagnostics["mean_nees"] > nees_threshold
    non_catastrophic_mask = ~catastrophic_mask

    # Total seeds for each scenario and fractions
    n_close_spawn = int(close_spawn_mask.sum())
    frac_close_spawn = float(n_close_spawn / n_total)
    n_catastrophic = int(catastrophic_mask.sum())
    frac_catastrophic = float(n_catastrophic / n_total)

    # Pooled NEES for all, excluding catastrophic, and median
    pooled_nees_all = float(np.nanmean(diagnostics["mean_nees"]))

    # Protect against all catastrophic
    if non_catastrophic_mask.any():
        pooled_nees_excl_catastrophic = float(np.nanmean(diagnostics["mean_nees"][non_catastrophic_mask]))
    else:
        pooled_nees_excl_catastrophic = np.nan

    median_per_seed_nees = float(np.nanmedian(diagnostics["mean_nees"]))

    # Outlier seeds
    outlier_seeds = diagnostics["seeds"][catastrophic_mask].tolist()

    outlier_dict = {"n_total": n_total, 
                    "n_close_spawn": n_close_spawn, 
                    "frac_close_spawn": frac_close_spawn, 
                    "n_catastrophic": n_catastrophic,
                    "frac_catastrophic": frac_catastrophic,
                    "pooled_nees_all": pooled_nees_all,
                    "pooled_nees_excl_catastrophic": pooled_nees_excl_catastrophic,
                    "median_per_seed_nees": median_per_seed_nees, 
                    "outlier_seeds": outlier_seeds}

    return outlier_dict


def plot_nees_vs_range(diagnostics: dict, save_path: Path, min_range_threshold: float, nees_threshold: float = 20.0) -> None:
    """Plot mean_nees vs min_range
    In: diagnostics, output of compute_per_seed_diagnostics;
        save_path, Path;
        min_range_threshold, config['scenarios']['...']['sigma_range']
        nees_threshold, per-seed mean NEES above this is flagged "catastrophic" (20 for now)
    """

    # Check if path exists
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Support passing either a flat diagnostics dict or a split_by_regime dict
    if "overall" in diagnostics:
        data = diagnostics["overall"]
    else:
        data = diagnostics

    min_range = np.asarray(data["min_range"], dtype=float)
    mean_nees = np.asarray(data["mean_nees"], dtype=float)

    # Filter out NaNs 
    valid = np.isfinite(min_range) & np.isfinite(mean_nees) & (mean_nees > 0)
    min_range = min_range[valid]
    mean_nees = mean_nees[valid]

    plt.figure(figsize=(12, 6))

    # Color points by whether they exceed the catastrophic NEES threshold
    is_catastrophic = mean_nees > nees_threshold
    plt.scatter(
        min_range[~is_catastrophic],
        mean_nees[~is_catastrophic],
        color="blue",
        label=f"Nominal (NEES <= {nees_threshold:g})",
    )

    if is_catastrophic.any():
        plt.scatter(
            min_range[is_catastrophic],
            mean_nees[is_catastrophic],
            color="red",
            label=f"Catastrophic (NEES > {nees_threshold:g})",
        )

    # Threshold reference lines to highlight the close-to-spawn regime
    plt.axhline(
        nees_threshold,
        color="red",
        linestyle="--",
        label=f"NEES threshold ({nees_threshold:g})",
    )

    plt.axvline(
        min_range_threshold,
        color="orange",
        linestyle="--",
        label=f"Min range threshold ({min_range_threshold:g})",
    )

    plt.yscale("log")
    plt.xlabel("Minimum Range to Sensor")
    plt.ylabel("Per-Seed Mean NEES (log)")
    plt.title("Per-Seed Mean NEES vs. Minimum Range")

    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def run_scenario_outlier_report(scenario_name: str, scenario_fn: Callable, build_filter_fn: Callable, config: dict, mc_seeds: list[int], min_range_threshold: float, split_by_regime: bool = False, methods: list[str] | None = None, sensor_pos: np.ndarray = np.zeros(2)) -> None:
    """ For each scenario and method get outlier summary and nees_vs_range plot
    In: scenario_fn, one of the functions from fusion.eval.scenarios;
        filter_builder_fn(trajectory, z_seq), has scenario specific knowledge (sigma_a);
        config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
        min_range_threshold, config['scenarios']['...']['sigma_range'];
        split_by_regime, True only for maneuvering's EKF/UKF/IMM rows
        methods, Optional (Only used for contaminated for UKF)
    """
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    figures_dir = project_root / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if methods is not None:
        methods = methods
    else:
        methods = config['scenarios'][scenario_name]['methods']

    for method in methods:
        filter_builder_fn = lambda traj, z: build_filter_fn(method, traj, z)

        stat_dict = compute_per_seed_diagnostics(scenario_fn, filter_builder_fn, config, mc_seeds, split_by_regime, sensor_pos)

        if split_by_regime:
            for regime in ("straight", "turning", "overall"):
                diag = stat_dict[regime]
                stem = f"{scenario_name}_{method}_{regime}"
                outlier_dict = summarize_outliers(diag, min_range_threshold)

                print(f"{stem}")
                print(json.dumps(outlier_dict, indent=2, allow_nan=True))
                print("")

                with open(processed_dir / f"{stem}_outliers.json", "w") as f:
                    json.dump(outlier_dict, f, indent=2, allow_nan=True)

                plot_nees_vs_range(diag, figures_dir / f"{stem}_nees_vs_range.png", min_range_threshold)
        else:
            stem = f"{scenario_name}_{method}"
            outlier_dict = summarize_outliers(stat_dict, min_range_threshold)

            print(f"{stem}")
            print(json.dumps(outlier_dict, indent=2, allow_nan=True))
            print("")

            with open(processed_dir / f"{stem}_outliers.json", "w") as f:
                json.dump(outlier_dict, f, indent=2, allow_nan=True)

            plot_nees_vs_range(stat_dict, figures_dir / f"{stem}_nees_vs_range.png", min_range_threshold)


def run_dual_sensor_fusion_outlier_report(config: dict, mc_seeds: list[int]) -> None:
    """ For each scenario and method get outlier summary and nees_vs_range plot
    In: config, final_comparison.yaml;
        mc_seeds, config['comparison']['mc_seeds'];
    """
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    figures_dir = project_root / "figures"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    min_range_threshold = config["scenarios"]["dual_sensor_fusion"]["sigma_range"]

    for filter_cls, filter_name in [(ExtendedKalmanFilter, "EKF"), (UnscentedKalmanFilter, "UKF")]:
        for fuse_style in ("sequential", "naive", "ci"):

            stat_dict = compute_per_seed_diagnostics_dual_sensor(filter_cls, fuse_style, config , mc_seeds)

            stem = f"dual_sensor_fusion_{fuse_style}_{filter_name}"
            outlier_dict = summarize_outliers(stat_dict, min_range_threshold)

            print(f"{stem}")
            print(json.dumps(outlier_dict, indent=2, allow_nan=True))
            print("")

            with open(processed_dir / f"{stem}_outliers.json", "w") as f:
                json.dump(outlier_dict, f, indent=2, allow_nan=True)

            plot_nees_vs_range(stat_dict, figures_dir / f"{stem}_nees_vs_range.png", min_range_threshold)

# Builder functions for run_scenario_outlier_report

# Benign
def benign_filter_fn(method, traj, z):
    sigma_a = config["tuned"]["benign"][method]

    if method == "EKF":
        filter_cls = ExtendedKalmanFilter
    elif method == "UKF":
        filter_cls = UnscentedKalmanFilter

    return build_benign_filter(sigma_a, traj, z, filter_cls, config)

# Manuevering
def maneuvering_filter_fn(method, traj, z):
    if method == "IMM":
        return build_imm_filter(traj, z, config)
    
    sigma_a = config["tuned"]["maneuvering"][method]

    if method == "EKF":
        filter_cls = ExtendedKalmanFilter
    elif method == "UKF":
        filter_cls = UnscentedKalmanFilter

    return build_maneuvering_filter(sigma_a, traj, z, filter_cls, config)

# Contaminated
def contaminated_filter_fn(method, traj, z):
    sigma_a = config["tuned"]["contaminated"][method]

    if method == "EKF":
        filter_cls = ExtendedKalmanFilter
    elif method == "UKF":
        filter_cls = UnscentedKalmanFilter

    return build_contaminated_filter(sigma_a, traj, z, filter_cls, config)

# Bearing only
def bearing_only_filter_fn(method, traj, z):
    sigma_a = config["tuned"]["bearing_only"][method]

    if method == "EKF":
        filter_cls = ExtendedKalmanFilter
    elif method == "UKF":
        filter_cls = UnscentedKalmanFilter

    return build_bearing_only_filter(sigma_a, traj, z, filter_cls, config)

def main() -> None:
    """
    Computes per seed diagnostics for rows
    Calls run_scenario_outlier for each scenario
    """

    # mc_seeds to use and min_range_threshold
    mc_seeds = config["comparison"]["mc_seeds"]

    # Run scenario outlier reports (min_range_threshold each respective sigma_a)
    run_scenario_outlier_report("benign", build_benign_scenario, benign_filter_fn, config, mc_seeds, config["scenarios"]["benign"]["sigma_range"])
    run_scenario_outlier_report("maneuvering", build_maneuvering_scenario, maneuvering_filter_fn, config, mc_seeds, config["scenarios"]["maneuvering"]["sigma_range"], split_by_regime=True)
    run_scenario_outlier_report("contaminated", build_contaminated_scenario, contaminated_filter_fn, config, mc_seeds, config["scenarios"]["contaminated"]["sigma_range"], methods=["EKF"])
    # Setting min_range_threshold to 20 (Arbitrary as no sigma_range for bearing_only)
    run_scenario_outlier_report("bearing_only", build_bearing_only_scenario, bearing_only_filter_fn, config, mc_seeds, 20, sensor_pos=np.array(config['scenarios']['bearing_only']['sensor_pos']))

    run_dual_sensor_fusion_outlier_report(config, mc_seeds)

    
if __name__ == "__main__": main()
