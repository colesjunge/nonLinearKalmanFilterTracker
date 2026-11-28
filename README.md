# Nonlinear Kalman Filter Tracker 

This project centers focuses on implementing nonlinear sensor fusion and learned signal processing, built on top of [multiTargetKalmanFilterTracker](https://github.com/colesjunge/multiTargetKalmanFilterTracker) (imported as a dependency).

This project focuses on relaxing the limited assumptions of the first project (linear movement, Gaussian noise, cartesian positions, etc.). It implements **Extended and Unscented Kalman Filters** for the nonlinear measurement models, **sensor fusion** (measurement-level sequential fusion as well as track-level naive and covariance intersection), and **learned components** (denoising, dynamics, clutter classification) that repair specific, identified failures of the classical baseline.

## Project Structure

```
nonlinear-fusion-tracking/
├── README.md
├── RESULTS.md
├── pyproject.toml                        
├── requirements.txt                      # pins target-tracking@v1.0 (Project 1) & this project's dependencies
├── configs/
│   ├── training.yaml                     # shared hyperparams for denoiser/covariance/dynamics/full-pipeline training runs
│   ├── imm.yaml                          # IMM mode set, per-mode process noise
│   └── final_comparison.yaml             # Final comparison harness; tuned sigma_a for each (method, scenario), dataset paths, mc_seeds
│
├── scripts/
│   │   # --- Stage demos ---
│   ├── demo_angle_wrapping.py            # EKF diverges/converges crossing ±π bearing w/o vs w/ wrapping
│   ├── demo_ukf_vs_ekf.py                # UKF vs EKF on a close-range, anisotropic-uncertainty scenario
│   ├── demo_sequential_fusion.py         # Radar + EO/IR sequential fusion, covariance-ellipse comparison
│   ├── demo_track_fusion.py              # Naive vs Covariance Intersection track-level fusion
│   ├── demo_contaminated_noise.py        # EKF/UKF degradation under contaminated-Gaussian sensor noise
│   │   # --- Dataset generation ---
│   ├── generate_dataset.py               # CV trajectories + contaminated radar noise -> denoiser windows
│   ├── generate_bearing_only_dataset.py  # bearing-only analogue of generate_dataset.py
│   ├── generate_maneuver_dataset.py      # waypoint (CV + coordinated-turn) trajectories
│   ├── generate_full_pipeline_dataset.py         # maneuvering + contaminated noise -> full-pipeline windows
│   ├── generate_full_pipeline_bearing_dataset.py # bearing-only analogue of generate_full_pipeline_dataset.py 
│   │   # --- Training ---
│   ├── train_denoiser.py                 # Causal denoiser trunk
│   ├── train_bearing_denoiser.py         # Bearing-only Causal denoiser trunk
│   ├── train_adaptive_r.py               # Per-sample-NLL CovarianceHead (frozen trunk)
│   ├── train_rollout_covariance.py       # CovarianceHead retrained through a DifferentiableKF rollout loss
│   ├── train_bearing_rollout_covariance.py # Bearing only CovarianceHead
│   ├── train_residual_dynamics.py        # ResidualDynamicsNet, trained through DifferentiableKF
│   ├── train_clutter_classifier.py       # HistGradientBoostingClassifier Clutter Classifier
│   ├── train_full_pipeline.py            # Joint denoiser+CovarianceHead+ResidualDynamicsNet
│   ├── train_full_pipeline_bearing.py    # Bearing-only Joint denoiser+CovarianceHead+ResidualDynamicsNet
│   │   # --- Tuning (brentq against NEES=4, seeds disjoint from comparison trials) ---
│   ├── tune_baseline_alpha.py            # Tuned scalar-alpha baseline for denoiser's induced overconfidence
│   ├── tune_maneuver_baseline.py         # sigma_a for CV/IMM/ResidualDynamicsKF, per regime
│   ├── tune_benign.py                    # per-method re-tune, benign column
│   ├── tune_contaminated.py              # per-method re-tune, contaminated column
│   ├── tune_maneuvering.py               # per-method re-tune, maneuvering column
│   ├── tune_bearing_only.py              # per-method re-tune, bearing-only column
│   ├── tune_dual_sensor_fusion.py        # per-(filter,fusion-style) re-tune, dual-sensor-fusion column
│   ├── tuning_common_fn.py               # shared brentq/NEES-objective helpers for the tune scripts
│   ├── sweep_birth_threshold.py          # sweeps the clutter classifier's track-birth threshold
│   │   # --- Comparisons, validation, and diagnostics ---
│   ├── verify_seam.py                    # reproduces Project 1's demo result through the pinned dependency
│   ├── run_comparison.py                 # denoiser / +learned-R / +rollout-retrain comparison
│   ├── run_maneuver_comparison.py        # CV vs IMM vs ResidualDynamicsKF on maneuvering data
│   ├── run_clutter_comparison.py         # kinematic-gate baseline vs. classifier-gated tracker
│   ├── run_final_comparison.py           # full 6-scenario by method comparison harness
│   ├── comparison_common_fn.py           # shared scenario/filter-building helpers for run_final_comparison.py
│   ├── validate_bearing_denoiser.py      # checks bearing-only denoiser converges before training full-pipeline-bearing
│   ├── recheck_clutter_calibration_ukf.py # confirms the clutter classifier's calibration on UKF features
│   └── report_scenario_outliers.py       # per-seed NEES vs min-range-to-sensor outlier diagnostics, all 6 columns
│
├── src/fusion/
│   ├── __init__.py
│   ├── models/                           # MeasurementModel subclasses
│   │   ├── radar.py                      # RadarRangeBearing
│   │   ├── eoir.py                       # EOIRBearingOnly
│   │   ├── adaptive_position.py          # AdaptivePosition2D, a settable-R MeasurementModel
│   │   └── motion.py                     # CoordinatedTurn2D, a MotionModel subclass
│   ├── filters/
│   │   ├── ekf.py                        # ExtendedKalmanFilter (w/ Joseph-form update)
│   │   ├── ukf.py                        # sigma_points, UnscentedKalmanFilter (w/ generalized Joseph form)
│   │   ├── imm.py                        # InteractingMultipleModel
│   │   └── utils.py                      # wrap_angle, residual_with_angles, gaussian_mixture_moments/likelihood
│   ├── fuse/
│   │   ├── sequential.py                 # fuse_sequential
│   │   ├── naive.py                      # fuse_naive
│   │   └── covariance_intersection.py    # covariance_intersection
│   ├── sim/
│   │   ├── noise.py                      # contaminated_gaussian_noise
│   │   ├── maneuvers.py                  # coordinated_turn_trajectory, waypoint_trajectory
│   │   └── sensors_nl.py                 # RadarSensor, Poisson clutter & free ground-truth labels
│   ├── learned/
│   │   ├── denoiser.py                   # Denoiser (causal dilated 1-D CNN)
│   │   ├── datasets.py                   # DenoiserDataset, RolloutWindowDataset, ManeuverRolloutDataset
│   │   ├── train.py                      # train_epoch / evaluate / train (shared training-loop functions)
│   │   ├── checkpoints.py                # save_checkpoint / load_checkpoint
│   │   ├── covariance.py                 # gaussian_nll, CovarianceHead, AdaptiveDenoiser, RolloutCovarianceModel
│   │   ├── diff_kf.py                    # DifferentiableKF, batched torch KF recursion
│   │   ├── dynamics.py                   # ResidualDynamicsNet
│   │   ├── residual_dynamics_kf.py       # ResidualDynamicsKF, build_residual_dynamics_kf
│   │   ├── clutter.py                    # compute_features, build_clutter_dataset, build_cost_matrix_with_classifier
│   │   ├── clutter_tracker.py            # MultiTargetTrackerClutter
│   │   └── full_pipeline.py              # FullPipelineModel & its two builder functions (range/bearing, bearing-only)
│   └── eval/
│       ├── filters.py                    # per-scenario filter builders for final comparison harness
│       ├── scenarios.py                  # per-scenario trajectory/measurement builders for final comparison harness
│       ├── metrics.py                    # false_track_rate, classify_maneuver_segments 
│       ├── rollout.py                    # run_filter_on_trajectory, generic single-target RMSE/NEES harness
│       └── viz.py                        # plot_covariance_ellipse
│
└── tests/
    ├── test_ekf.py                       # EKF vs. Project 1's KF on a linear h; radar Jacobian vs. numerical check
    ├── test_angles.py                    # wrap_angle / residual_with_angles correctness
    ├── test_ukf.py                       # sigma_points reconstruction; UKF vs. KF on a linear h; Joseph-form PD check
    ├── test_imm.py                       # InteractingMultipleModel mixing/combination
    ├── test_fusion.py                    # fuse_sequential order-invariance, covariance reduction
    ├── test_naive_fusion.py              # fuse_naive overconfidence under correlated tracks
    ├── test_covariance_intersection.py   # CI never more confident than naive fusion
    ├── test_contaminated_noise.py        # contaminated_gaussian_noise's per-sample Bernoulli branch behavior
    ├── test_adaptive_position.py         # AdaptivePosition2D's settable R
    ├── test_learned_shapes.py            # Denoiser causality (test_causality) and input/output shapes
    ├── test_learned_covariance.py        # gaussian_nll, CovarianceHead PSD/Cholesky guarantees
    ├── test_diff_kf.py                   # DifferentiableKF vs. NumPy KF (1e-6), symmetry, gradient flow, batch independence
    ├── test_dynamics.py                  # ResidualDynamicsNet output shapes and Cholesky-parameterized noise
    ├── test_residual_dynamics_kf.py      # ResidualDynamicsKF end-to-end
    ├── test_coordinated_turn.py          # coordinated_turn_trajectory closed-form arc correctness
    ├── test_waypoint_trajectory.py       # waypoint_trajectory leg-chaining
    └── test_full_pipeline.py             # FullPipelineModel forward pass shapes
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Headline Results

Below is the final comparison matrix, methods (EKF, UKF, IMM, +denoiser, +denoiser+learned R, +learned dynamics, full pipeline, six fusion-style rows) and scenarios (benign, maneuvering, contaminated noise, heavy clutter, bearing-only, dual-sensor fusion).

The values are **RMSE / NEES**, while for the heavy clutter column, **ID switches / false-track rate**.

**For all findings across every stage please see [RESULTS.md](RESULTS.md)**

*Several cells are not direct comparisons* (different noise scales, different measurement models, different trajectory ensembles between rows in the same column). Full explanations of every footnote, plus the per-seed outlier diagnostics are explained in in [RESULTS.md](RESULTS.md#stage-13-final-comparison-matrix-methods--scenarios-rmse--nees).

| Method | Benign | Maneuvering (overall) | Contaminated | Heavy Clutter | Bearing-Only | Dual-Sensor Fusion |
|---|---|---|---|---|---|---|
| EKF | 10.47 / 38.34 ¹ | 28.04 / 123.65 ¹ | 22.06 / 246.62 ² | 1.667 / 0.165 ⁹ | 8475.85 / 208.49 ² | — |
| UKF | 13.36 / 13.42 ¹ | 9.01 / 4.14 ¹ | — ³ | 1.667 / 0.165 ⁴ ⁹ | 307.47 / 832.0 ² | — |
| IMM | — | 15.92 / 81.14 ¹ | — | — | — | — |
| + denoiser | — | — | 28.10 / 24.77 ⁵ | — | 56.02 / 8.00 ⁵ | — |
| + denoiser/learned R | — | — | 30.10 / 7.31 ⁵ | — | 57.58 / 0.68 ⁵ | — |
| + learned dynamics | — | 3.94 / 5.44 ⁶ | — | — | 3.94 / 5.44 ⁶ | — |
| Full pipeline | — | 31.89 / 4.18 ⁷ | 46.65 / 3.91 ⁷ | — | 67.57 / 5.28 ⁷ | — |
| Sequential fusion (EKF) | — | — | — | — | — | 5.93 / 3.33 |
| Sequential fusion (UKF) | — | — | — | — | — | 18.08 / 53.24 ⁸ |
| Naive fusion (EKF) | — | — | — | — | — | 6.19 / 159.11 ⁸ |
| Naive fusion (UKF) | — | — | — | — | — | 32.24 / 158.20 ⁸ |
| CI fusion (EKF) | — | — | — | — | — | 7.24 / 117.63 ⁸ |
| CI fusion (UKF) | — | — | — | — | — | 32.53 / 90.49 ⁸ |

## On AI Assistance

I used Claude Code in this project explicitly as a debugging and testing guide. It did not write any code, but it was useful for reviewing and debugging files to ensure that all functions were correctly implemented and integrated. Rather than fixing issues, I had it explain their root causes, allowing me to make and learn the appropriate fixes myself.

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
