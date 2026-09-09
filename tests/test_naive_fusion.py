import numpy as np
from fusion.fuse.naive import fuse_naive

def test_fuse_naive_equal_covariances_averages():
    """
    Test that equal covariances (P1 = P2) result in P1/2
    Test that states (x1, x2) result in (x1+x2)/2
    """

    x1 = np.array([95, 40, 5, 10])
    x2 = np.array([5, 20, 25, -2])

    P1 = np.array([[10, 0, 0, 0], 
                    [0, 12, 0, 0], 
                    [0, 0, 34, 0], 
                    [0, 0, 0, 35]])
    P2 = P1.copy()

    x_fused, P_fused = fuse_naive(x1, P1, x2, P2)

    # Check whether equal covariances result in average as well as x_fused
    assert np.allclose(P_fused, (P1)/2, atol=1e-10), "Covariance arithmetic incorrect"
    assert np.allclose(x_fused, (x1+x2)/2, atol=1e-10), "State arithmetic incorrect"

def test_fuse_naive_order_invariant():
    """
    Test order invariance
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

    x_fused_1, P_fused_1 = fuse_naive(x1, P1, x2, P2)
    x_fused_2, P_fused_2 = fuse_naive(x2, P2, x1, P1)

    # Check if the updated measurements are the same regardless of the order
    assert np.allclose(x_fused_1, x_fused_2, atol=1e-10), "States do not match"
    assert np.allclose(P_fused_1, P_fused_2, atol=1e-10), "Covariances do not match"

def test_fuse_naive_reduces_covariance():
    """
    Test whether fusion reduces covariance
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

    _, P_fused = fuse_naive(x1, P1, x2, P2)

    det_P1 = np.linalg.det(P1)
    det_P2 = np.linalg.det(P2)
    det_P_fused = np.linalg.det(P_fused)



    # Check if fused covariance is less than individual covariances
    assert det_P_fused < det_P1, "Covariance P1 is less than fused"
    assert det_P_fused < det_P2, "Covariance P2 is less than fused"

# To run: pytest tests/test_naive_fusion.py -v