import numpy as np
from scipy import linalg

from tracking.filters.base import BayesFilter


def run_filter_on_trajectory(filter: BayesFilter, trajectory: np.ndarray, z_seq: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """ Helper function to be used to run running filters on trajectories
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










