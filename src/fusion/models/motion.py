import numpy as np

from tracking.models.motion import MotionModel

class CoordinatedTurn2D(MotionModel):
    """ 
    Motion model that stores fixed turn rate omega and sigma_a
    """

    dim_x = 4

    def __init__(self, omega: float, sigma_a: float) -> None:

        self.omega = omega
        self.sigma_a = sigma_a

        if self.omega == 0:
            raise ValueError("Omega must not be 0")

    def f(self, x: np.ndarray, dt: float) -> np.ndarray:
        """ Applies closed form CT transition matrix to x
        In: x (4, ), current state;
            dt, seconds since last update
        Out: pred_x (4, ), predicted state
        """

        transition = self.F(x, dt)
        pred_x = transition @ x

        return pred_x

    def F(self, x: np.ndarray, dt: float) -> np.ndarray:
        """ Returns transition matrix used in f
        In: x (4, ), current state (ignored);
            dt, seconds since last update
        Out: transition (4, 4), transition matrix
        """

        velocity_update = np.array([
                                [np.cos(self.omega*dt), -np.sin(self.omega*dt)],
                                [np.sin(self.omega*dt), np.cos(self.omega*dt)]
                                ])
    
        position_update = np.array([
                            [np.sin(self.omega*dt), np.cos(self.omega*dt) - 1],
                            [1 - np.cos(self.omega*dt), np.sin(self.omega*dt)]
                            ])
    
        transition = np.block([
                            [np.eye(2), (1/self.omega)* position_update],
                            [np.zeros((2,2)), velocity_update]
                            ])

        return transition

    def Q(self, dt: float) -> np.ndarray:
        """
        Calculates process noise covariance 
        In: dt, seconds since last update
        Out: noise_cov (4,4)
        """

        noise_cov = np.array([
                            [dt**4/4, 0, dt**3/2, 0],
                            [0, dt**4/4, 0, dt**3/2],
                            [dt**3/2, 0, dt**2, 0],
                            [0, dt**3/2, 0, dt**2]
                          ]) * self.sigma_a**2
        
        return noise_cov