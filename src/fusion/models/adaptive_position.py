import numpy as np
from tracking.models.measurement import MeasurementModel

class AdaptivePosition2D(MeasurementModel):
    """
    Same measurement space as LinearPosition2D (px, py), but R is set
    externally per-step from a learned covariance, not fixed at construction.
    """
    dim_z = 2

    def __init__(self, R_init: np.ndarray):
        self._R = R_init

    def h(self, x: np.ndarray) -> np.ndarray:
        """Predicted measurement given state.
        In:  x shape (4,); current state estimate
        Out: shape (2,); predicted measurement
        """
        selection_matrix = self.H(x)
        pred_z = selection_matrix @ x

        return pred_z

    def H(self, x: np.ndarray) -> np.ndarray:
        """Measurement Jacobian.
        In:  x shape (4,); current state estimate
        Out: shape (2, 4); Jacobian of h w.r.t. x, evaluated at x
        """

        selection_matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        return selection_matrix
    def R(self) -> np.ndarray:
        """
        Return stored R
        """
        return self._R
    
    def set_R(self, R: np.ndarray) -> None:
        """
        Update stored R
        """
        self._R = R
