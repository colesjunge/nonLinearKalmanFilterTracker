import numpy as np
from tracking.models.measurement import MeasurementModel

class EOIRBearingOnly(MeasurementModel):
    """
    Measurement model for EOIR bearing-only measurements.
    """

    dim_z = 1 # Measurement dimension (bearing only)
    angle_indices = [0] # The index of the angle in the measurement vector

    def __init__(self, sigma_bearing: float, sensor_pos: np.ndarray | None = None):
        self.sigma_bearing = sigma_bearing # Radians
        self.sensor_pos = sensor_pos if sensor_pos is not None else np.zeros(2) # Sensor position in global coordinates

    def h(self, x: np.ndarray) -> np.ndarray:
        """ Measurement function for EOIR bearing-only measurements.
        In: x; state vector (4,)
        Out: Measurement vector (1,)
        """
        px, py, _, _ = x
        dx = px - self.sensor_pos[0]
        dy = py - self.sensor_pos[1]
        bearing = np.arctan2(dy, dx)

        return np.array([bearing])

    def H(self, x: np.ndarray) -> np.ndarray:
        """ Jacobian of the measurement function.
        In: x; state vector (4,)
        Out: Jacobian matrix (1, 4)
        """
        px, py, _, _ = x
        dx = px - self.sensor_pos[0]
        dy = py - self.sensor_pos[1]
        q = max(1e-6, dx**2 + dy**2)

        H = np.zeros((1, 4))
        H[0, 0] = -dy / q
        H[0, 1] = dx / q

        return H

    def R(self) -> np.ndarray:
        """ Returns the measurement noise covariance matrix for the EOIR bearing-only measurements.
        In: None
        Out: Measurement noise covariance matrix (1, 1)
        """
        return np.array([[self.sigma_bearing**2]])

