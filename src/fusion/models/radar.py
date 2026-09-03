import numpy as np
from tracking.models.measurement import MeasurementModel


class RadarRangeBearing(MeasurementModel):
    """ 
    Measurement model for radar range and bearing measurements.
    """

    dim_z = 2 # Measurement dimension (range and bearing)
    angle_indices = [1] # The index of the angle in the measurement vector

    def __init__(self, 
                 sigma_range: float, 
                 sigma_bearing: float, 
                 sensor_pos: np.ndarray | None = None):

        self.sigma_range = sigma_range # Meters
        self.sigma_bearing = sigma_bearing # Radians
        self.sensor_pos = sensor_pos if sensor_pos is not None else np.zeros(2) # Sensor position in global coordinates

    def h(self, x: np.ndarray) -> np.ndarray:
        """ Measurement function for radar range and bearing measurements.
        In: x; state vector (4,)
        Out: Measurement vector (2,)
        """

        px, py, _, _ = x
        dx = px - self.sensor_pos[0]
        dy = py - self.sensor_pos[1]
        q = max(1e-6, dx**2 + dy**2)
        range_ = np.sqrt(q)
        bearing = np.arctan2(dy, dx)

        return np.array([range_, bearing])


    def H(self, x: np.ndarray) -> np.ndarray:
        """ Jacobian of the measurement function.
        In: x; state vector (4,)
        Out: Jacobian matrix (2, 4)
        """
        px, py, _, _ = x
        dx = px - self.sensor_pos[0]
        dy = py - self.sensor_pos[1]
        q = max(1e-6, dx**2 + dy**2)
        range_ = np.sqrt(q)

        H = np.zeros((2, 4))

        H[0, 0] = dx / range_
        H[0, 1] = dy / range_
        H[1, 0] = -dy / q
        H[1, 1] = dx / q

        return H

    def R(self) -> np.ndarray:
        """ Returns the measurement noise covariance matrix for the radar range and bearing measurements.
        In: None
        Out: Measurement noise covariance matrix (2, 2)
        """

        return np.diag([self.sigma_range**2, self.sigma_bearing**2])
      


    