import numpy as np

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