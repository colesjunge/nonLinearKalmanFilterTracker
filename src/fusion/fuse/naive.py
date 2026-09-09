import numpy as np
from scipy import linalg

def fuse_naive(x1: np.ndarray, 
                P1: np.ndarray, 
                x2: np.ndarray,
                P2: np.ndarray,) -> tuple[np.ndarray, np.ndarray]:
    """
    Fuse estimates through inverse covariance weighting
    It should be noted that this is not optimal
    This can only be optimal if the two state estimates are independent
    In this case they share common process noise history, and thus this can be overconfident

    In: x1 (n,), P1 (n,n), x2 (n,), P2 (n,n); Two state estimates on same target
    Out: x_fused (n,), P_fused (n,n); New fused state estimate
    
    """

    I = np.eye(P1.shape[0])
    
    P1_inv = linalg.solve(P1, I)
    P2_inv = linalg.solve(P2, I)

    P_fused = linalg.solve(P1_inv + P2_inv, I)

    x_fused = P_fused @ ((P1_inv @ x1) + (P2_inv @ x2))

    return x_fused, P_fused

