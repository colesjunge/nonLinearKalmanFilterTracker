import numpy as np
from tracking.filters.utils import wrap_angle, residual_with_angles
import math


def test_wrap_angle_boundaries():
    """
    Test that wrap_angle correctly wraps angles 
    """

    # Assert that wrap_angle correctly wraps angles within the range (-pi, pi]
    assert np.isclose(wrap_angle(np.pi + 0.1) - (-np.pi + 0.1), 0), "wrap_angle failed to wrap angle greater than pi"

    # Assert that wrap_angle correctly wraps 3pi to pi
    assert np.isclose(wrap_angle(3*np.pi) - np.pi, 0), "wrap_angle failed to wrap angle greater than 2*pi"


def test_residual_across_branch_cut():
    """
    Test that residual_with_angles correctly computes the residual across the branch cut at pi and -pi.
    """

    residual = residual_with_angles(
        np.array([math.radians(179)]),
        np.array([math.radians(-179)]),
        [0]
    )

    assert np.isclose(residual, math.radians(-2)), "residual_with_angles failed across branch cut"

# To run: pytest tests/test_angles.py -v