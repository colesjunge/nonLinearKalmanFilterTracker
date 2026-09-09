import numpy as np
from scipy import linalg
from fusion.fuse.covariance_intersection import covariance_intersection
from fusion.fuse.naive import fuse_naive

def test_ci_omega_zero_returns_second_estimate():
    """
    Test that when omega is set to 0, P2 is returned
    """

    x1 = np.array([95, 40, 5, 10])
    x2 = np.array([5, 20, 25, -2])

    P1 = np.array([[10, 0, 0, 0], 
                    [0, 12, 0, 0], 
                    [0, 0, 34, 0], 
                    [0, 0, 0, 35]])
    P2 = np.array([[20, 0, 0, 0], 
                        [0, 2, 0, 0], 
                        [0, 0, 13, 0], 
                        [0, 0, 0, 27]])

    omega = 0

    _, P_fused = covariance_intersection(x1, P1, x2, P2, omega)

    # Check whether fused covariance is equal to P2
    assert np.allclose(P_fused, P2, atol=1e-10), "P_fused not equal to P2"

def test_ci_omega_one_returns_first_estimate():
    """
    Test that when omega is set to 1, P1 is returned
    """

    x1 = np.array([95, 40, 5, 10])
    x2 = np.array([5, 20, 25, -2])

    P1 = np.array([[10, 0, 0, 0], 
                    [0, 12, 0, 0], 
                    [0, 0, 34, 0], 
                    [0, 0, 0, 35]])
    P2 = np.array([[20, 0, 0, 0], 
                        [0, 2, 0, 0], 
                        [0, 0, 13, 0], 
                        [0, 0, 0, 27]])

    omega = 1

    _, P_fused = covariance_intersection(x1, P1, x2, P2, omega)

    # Check whether fused covariance is equal to P1
    assert np.allclose(P_fused, P1, atol=1e-10), "P_fused not equal to P1"

def test_ci_equal_covariances_omega_half():
    """
    Test that equal covariances (P1 = P2) result in P1 when omega = .5
    """

    x1 = np.array([95, 40, 5, 10])
    x2 = np.array([5, 20, 25, -2])

    P1 = np.array([[10, 0, 0, 0], 
                    [0, 12, 0, 0], 
                    [0, 0, 34, 0], 
                    [0, 0, 0, 35]])
    P2 = P1.copy()

    omega = 0.5

    _, P_fused_ci = covariance_intersection(x1, P1, x2, P2, omega)

    # Check whether equal covariances result in average and result in same 
    assert np.allclose(P_fused_ci, P1, atol=1e-10), "Covariance arithmetic incorrect"

def test_ci_never_more_confident_than_naive():
    """
    Test that optimized covariance intersection is never more confident than naive fusion
    This is best tested with several x and P, thus they are randomly generated over ten trials
    """

    rng = np.random.default_rng(42)

    for _ in range(10):
        A = rng.normal(size=(4, 4))
        B = rng.normal(size=(4, 4))

        P1 = A @ A.T + np.eye(4) # To ensure symmetric positive definite
        P2 = B @ B.T + np.eye(4) # To ensure symmetric positive definite

        x1 = rng.normal(size=4)
        x2 = rng.normal(size=4)

        _, P_fused_ci = covariance_intersection(x1, P1, x2, P2)
        _, P_fused_naive = fuse_naive(x1, P1, x2, P2)

        assert np.linalg.det(P_fused_ci) >= np.linalg.det(P_fused_naive), "Naive is more confident than ci"

def test_ci_omega_minimizes_determinant():
    """
    Test that the omega selected by covariance intersection approximately minimizes the determinant
    """

    rng = np.random.default_rng(42)

    A = rng.normal(size=(4, 4))
    B = rng.normal(size=(4, 4))

    P1 = A @ A.T + np.eye(4)
    P2 = B @ B.T + np.eye(4)

    x1 = rng.normal(size=4)
    x2 = rng.normal(size=4)

    # Get optimizer's omega and covariance
    _, P_fused_ci = covariance_intersection(x1, P1, x2, P2)

    # Grid search
    omegas = np.linspace(0, 1, 1000)
    determinants = []

    I = np.eye(P1.shape[0])
        
    P1_inv = linalg.solve(P1, I)
    P2_inv = linalg.solve(P2, I)

    for omega in omegas:
        P_fused = linalg.solve(omega * P1_inv + (1 - omega) * P2_inv, I)
        determinants.append(np.linalg.det(P_fused))

    grid_min = min(determinants)
    optimizer_det = np.linalg.det(P_fused_ci)

    assert np.isclose(optimizer_det, grid_min, rtol=1e-3), "Optimized omega is not true minimizer"

# To run: pytest tests/test_covariance_intersection.py -v



