import numpy as np
from tracking.filters.kalman import KalmanFilter
from scipy import linalg


class ExtendedKalmanFilter(KalmanFilter):

     
    def update(self, z):
        """ Update the state with a new measurement z.
        Uses Joseph form for P and symmetrizes after as a correction for numerical stability.
        In: z shape (dim_z,). 
        Mutates self.x, self.P in place. Returns nothing.
        """

        innovation_res, innovation_cov = self.innovation(z)
        selection_matrix = self.measurement.H(self.x)
        kalman_gain = linalg.solve(innovation_cov, selection_matrix @ self.P).T

        self.x = self.x + kalman_gain @ innovation_res

        I_KH = np.eye(self.motion.dim_x) - kalman_gain @ selection_matrix

        # Joseph form for P update
        self.P = (I_KH @ self.P @ I_KH.T + (kalman_gain @ self.measurement.R() @ kalman_gain.T))
        
        self.P = (self.P + self.P.T) / 2  # Corrects roundoff asymmetry