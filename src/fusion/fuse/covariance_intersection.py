import numpy as np
from scipy import optimize, linalg


def covariance_intersection(x1: np.ndarray, 
                            P1: np.ndarray, 
                            x2: np.ndarray,
                            P2: np.ndarray, 
                            omega: float | None=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuse measurements through covariance intersection
    This does not assume measurements are independent and is thus never overconfident
    However, when correlation is low, it can be more conservative than the naive approach
    If omega is not provided, it is chosen to minimize the determinant of the fused covariance

    In: x1 (n,), P1 (n,n), x2 (n,), P2 (n,n); Two state estimates on same target
        omega float in [0,1] or None; controls how much weight each state is given
    Out: x_fused (n,), P_fused (n,n); New fused state estimate
    
    """

    I = np.eye(P1.shape[0])

    P1_inv = linalg.solve(P1, I)
    P2_inv = linalg.solve(P2, I)

    if omega is None:

        # Helper function to determine the when det(P) is minimized
        def objective(omega):
            P_fused = linalg.solve(omega*P1_inv + (1 - omega)*P2_inv, I)
            return np.linalg.det(P_fused)

        result = optimize.minimize_scalar(objective, bounds=(0, 1), method="bounded")

        omega = result.x

    P_fused = linalg.solve(omega*P1_inv + (1 - omega)*P2_inv, I)
    x_fused = P_fused @ (omega*(P1_inv @ x1) + (1-omega)*(P2_inv @ x2))

    return x_fused, P_fused

