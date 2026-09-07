import numpy as np
from tracking.filters.kalman import KalmanFilter
from scipy import linalg
from fusion.filters.utils import circular_mean
from tracking.filters.utils import residual_with_angles


def sigma_points(x, P, alpha, beta, kappa) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate sigma points and weights for the Unscented Kalman Filter.
    
    In: x (np.ndarray), state vector; P (np.ndarray), state covariance matrix; alpha, beta, kappa (float), UKF parameters
    Out: points (np.ndarray), sigma points; Wm (np.ndarray), weights for mean; Wc (np.ndarray), weights for covariance
    """
    n = x.shape[0]
    lambda_ = alpha**2 * (n + kappa) - n

    # Calculate square root of (n + lambda) * P
    sqrt_P = linalg.cholesky((n + lambda_) * P)

    # Generate sigma points
    points = np.zeros((2 * n + 1, n))
    points[0] = x
    for i in range(n):
        points[i + 1] = x + sqrt_P[i]
        points[n + i + 1] = x - sqrt_P[i]

    # Weights for mean and covariance
    Wm = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    Wc = np.full(2 * n + 1, 1 / (2 * (n + lambda_)))
    Wm[0] = lambda_ / (n + lambda_)
    Wc[0] = lambda_ / (n + lambda_) + (1 - alpha**2 + beta)

    return points, Wm, Wc


class UnscentedKalmanFilter(KalmanFilter):

    def __init__(self, motion_model, measurement_model, x0=None, P0=None, alpha=1e-3, beta=2.0, kappa=0.0):
        super().__init__(motion_model, measurement_model, x0=x0, P0=P0)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    # Predict will be inherited as motion model is linear at the moment, but this will be addressed later

    def _measurement_prediction(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict the measurement and its covariance using the Unscented Transform

        Out: z_pred (np.ndarray), predicted measurement; S (np.ndarray), innovation covariance; Cxz (np.ndarray), cross-covariance between state and measurement
        """
        # Generate sigma points
        points, Wm, Wc = sigma_points(self.x, self.P, self.alpha, self.beta, self.kappa)

        # Transform sigma points through the measurement model
        Z_sigma = np.array([self.measurement.h(point) for point in points]) # Shape (2n+1, dim_z)

        # Compute predicted measurement mean
        z_pred = np.zeros(Z_sigma.shape[1])
        for i in range(Z_sigma.shape[1]):
            if i in self.measurement.angle_indices:
                z_pred[i] = circular_mean(Z_sigma[:, i], Wm)
            else:
                z_pred[i] = np.sum(Wm * Z_sigma[:, i])

        # Compute innovation covariance
        S = np.zeros((Z_sigma.shape[1], Z_sigma.shape[1]))
        for i in range(Z_sigma.shape[0]):
            diff = residual_with_angles(Z_sigma[i], z_pred, self.measurement.angle_indices)
            S += Wc[i] * np.outer(diff, diff)
        
        S += self.measurement.R()  # Add measurement noise

        # Compute cross-covariance between state and measurement
        Cxz = np.zeros((self.motion.dim_x, Z_sigma.shape[1]))
        for i in range(points.shape[0]):
            diff_x = points[i] - self.x
            diff_z = residual_with_angles(Z_sigma[i], z_pred, self.measurement.angle_indices)
            Cxz += Wc[i] * np.outer(diff_x, diff_z)

        return z_pred, S, Cxz

    def innovation(self, z) -> tuple[np.ndarray, np.ndarray]:
        """Compute the innovation and its covariance for a given measurement z.
        
        In: z (np.ndarray), measurement vector (The ABC class for filters can only hanfle y and S, so Cxz must come from _measurement_prediction)
        Out: innovation (np.ndarray), innovation vector; innovation_cov (np.ndarray), innovation covariance matrix
        """
        z_pred, S, _ = self._measurement_prediction()
        y = z - z_pred

        # Wrap angles in the innovation
        y = residual_with_angles(z, z_pred, self.measurement.angle_indices)

        return y, S

    def update(self, z) -> None:
        """Update the state with a new measurement z using the Unscented Kalman Filter equations.
        
        In: z (np.ndarray), measurement vector
        Out: None; Mutates self.x, self.P in place
        """
        y, S = self.innovation(z)
        _, _, Cxz = self._measurement_prediction()
        kalman_gain = linalg.solve(S, Cxz.T).T  # K = Cxz * S^-1

        self.x = self.x + kalman_gain @ y
        self.P = self.P - kalman_gain @ S @ kalman_gain.T
        self.P = (self.P + self.P.T) / 2

