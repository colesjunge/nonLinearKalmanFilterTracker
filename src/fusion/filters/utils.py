import numpy as np
import scipy.linalg as linalg

def circular_mean(angles: np.ndarray, weights: np.ndarray) -> float:
    """
    Compute the circular mean of a set of angles (in radians) with given weights.
    
    In: angles (np.ndarray), array of angles in radians; weights (np.ndarray), array of weights corresponding to each angle 
    Out: float: The circular mean of the angles.
    """
    # Ensure angles are in the range [0, 2*pi)
    angles = np.mod(angles, 2 * np.pi)
    
    # Compute weighted sum of sine and cosine components
    sin_sum = np.sum(weights * np.sin(angles))
    cos_sum = np.sum(weights * np.cos(angles))
    
    # Compute the circular mean
    mean_angle = np.arctan2(sin_sum, cos_sum)
    
    # Ensure the mean angle is in the range [0, 2*pi)
    return np.mod(mean_angle, 2 * np.pi)


def gaussian_likelihood(y: np.ndarray, S: np.ndarray) -> float:
    """
    Evaluates N(y; 0, S)
    In: y (dim_z,) innovation;
        S (dim_z, dim_z) innovation covariance
    Out: likelihood (not log-liklihood)
    """

    dim_z = len(y)

    # Calculate log determinant of S
    sign, log_det = np.linalg.slogdet(S)

    if sign <= 0:
        raise ValueError("Innovation covariance must be positive definite")

    # Calculate squared Mahalanobis distance
    mahalanobis_sq = linalg.solve(S, y).dot(y)

    # Evaluate Gaussian density
    likelihood = (1 / np.sqrt((2 * np.pi) ** dim_z * np.exp(log_det))* np.exp(-0.5 * mahalanobis_sq))

    return float(likelihood)

def gaussian_mixture_moments(states: list[tuple[np.ndarray, np.ndarray]], weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Weighted mean of states plus the spread correction cross term added into weighted-P term
    Used in both IMM mixing and combination step
    In: states, N pairs of (x_i (dim_x,), P_i (dim_x, dim_x)); 
        weights (N,), sums to 1
    Out: x_comb (dim_x,), Combined states;
        P_comb (dim_x, dim_x)), Combined covariances
    """

    # Extract the number of models and state dimension
    N = len(states)
    dim_x = states[0][0].shape[0]

    # Initialize combined state and covariance
    x_comb = np.zeros(dim_x)
    P_comb = np.zeros((dim_x, dim_x))

    # Calculate weighted mean of the states
    for i in range(N):
        x_i, P_i = states[i]
        x_comb += weights[i] * x_i

    # Calculate weighted covariance including spread correction
    for i in range(N):
        x_i, P_i = states[i]

        # Difference between individual and combined estimates
        dx = x_i - x_comb

        # Add weighted covariance and spread correction
        P_comb += weights[i] * (P_i + np.outer(dx, dx))

    return x_comb, P_comb