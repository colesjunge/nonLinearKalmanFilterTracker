import numpy as np
from scipy import linalg
from typing import Callable

from tracking.filters.base import BayesFilter
from tracking.models.measurement import MeasurementModel
from fusion.fuse.sequential import fuse_sequential



def run_filter_on_trajectory(filter: BayesFilter, trajectory: np.ndarray, z_seq: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """ Helper function to be used to run filters on trajectories
    In: filter (works with both KalmanFilter and InteractingMultipleModel);
       trajectory (T, 4), ground truth;
       z_seq (T, 2), pre-generated noisy position measurements;
       dt, seconds since last update
    Out: x_est_seq (T, 4), filter.x after every step;
        nees_seq (T,), per-step NEES
    """

    T = trajectory.shape[0]

    # Empty outputs 
    x_est_seq = np.zeros((T,4))
    nees_seq = np.zeros(T)

    for t in range(T):
        # Predict step
        filter.predict(dt)
        # Update step
        filter.update(z_seq[t])

        # Update estimation states
        x_est_seq[t] = filter.x

        # Update NEES (error^T @ P^-1 @ error)
        error = trajectory[t] - filter.x
        nees_seq[t] = error @ linalg.solve(filter.P, error)

    return (x_est_seq, nees_seq)

def run_sequential_fusion_on_trajectory(filter: BayesFilter, 
                                        trajectory: np.ndarray, 
                                        z_radar_seq: np.ndarray, 
                                        z_eoir_seq: np.ndarray, 
                                        dt: float, 
                                        radar_model: MeasurementModel, 
                                        eoir_model: MeasurementModel) -> tuple[np.ndarray, np.ndarray]:
    """ Helper function to be used to run sequential fusion on trajectories
    In: filter (works with both KalmanFilter and InteractingMultipleModel);
       trajectory (T, 4), ground truth;
       z_radar_seq (T, 2), pre-generated noisy range-bearing measurements;
       z_eoir_seq (T, 1), pre-generated noisy bearing only measurements;
        dt, seconds since last update
       radar_model, RadarRangeBearing;
       eoir_model, EOIRBearingOnly;
    Out: x_est_seq (T, 4), filter.x after every step;
        nees_seq (T,), per-step NEES
    """

    T = trajectory.shape[0]

    # Empty outputs 
    x_est_seq = np.zeros((T,4))
    nees_seq = np.zeros(T)

    for t in range(T):
        # Predict step
        filter.predict(dt)

        # Fuse and Update step
        fuse_sequential(filter, [(radar_model, z_radar_seq[t]), (eoir_model, z_eoir_seq[t])])

        # Update estimation states
        x_est_seq[t] = filter.x

        # Update NEES (error^T @ P^-1 @ error)
        error = trajectory[t] - filter.x
        nees_seq[t] = error @ linalg.solve(filter.P, error)

    return (x_est_seq, nees_seq)

def run_track_level_fusion_on_trajectory(filter_radar: BayesFilter, 
                                        filter_eoir: BayesFilter,
                                        trajectory: np.ndarray, 
                                        z_radar_seq: np.ndarray, 
                                        z_eoir_seq: np.ndarray, 
                                        dt: float, 
                                        fuse_fn: Callable) -> tuple[np.ndarray, np.ndarray]:
    """ Helper function to be used to run track fusion on trajectories
    In: filt_radar & filt_eoir, two independent filters sharing the same initial prior;
       trajectory (T, 4), ground truth;
       z_radar_seq (T, 2), pre-generated noisy range-bearing measurements;
       z_eoir_seq (T, 1), pre-generated noisy bearing only measurements;
       dt, seconds since last update
       fuse_fn(x1, P1, x2, P2) -> (x_fused, P_fused), either fuse_naive or covariance_intersection
    Out: x_fused_seq (T, 4), filter.x after every step (fused);
        nees_seq (T,), per-step NEES
    """

    T = trajectory.shape[0]

    # Empty outputs 
    x_fused_seq = np.zeros((T,4))
    nees_seq = np.zeros(T)

    for t in range(T):

        # Predict and Update steps
        filter_radar.predict(dt)
        filter_radar.update(z_radar_seq[t])
        filter_eoir.predict(dt)
        filter_eoir.update(z_eoir_seq[t])

        # Fuse using fuse_fn
        x_fused, P_fused = fuse_fn(filter_radar.x, filter_radar.P, filter_eoir.x, filter_eoir.P)
        
        # Update estimation states
        x_fused_seq[t] = x_fused

        # Update NEES (error^T @ P^-1 @ error)
        error = trajectory[t] - x_fused
        nees_seq[t] = error @ linalg.solve(P_fused, error)

    return (x_fused_seq, nees_seq)

