import numpy as np
import pytest

from fusion.models.motion import CoordinatedTurn2D
from tracking.sim.trajectories import coordinated_turn_trajectory

def test_f_matches_trajectory_generator():
    """
    Verify that repeated motion-model matches the independently implemented coordinated-turn trajectory 
    """

    omega = 3.0
    dt = 0.1
    N = 100

    motion = CoordinatedTurn2D(omega=omega, sigma_a=0)

    x0 = np.array([0.0, 0.0, 10.0, 5.0])

    # Generate trajectory using the motion model
    model_trajectory = [x0.copy()]
    x = x0.copy()

    for _ in range(N-1):
        x = motion.f(x, dt)
        model_trajectory.append(x.copy())

    model_trajectory = np.asarray(model_trajectory)

    # Generate trajectory using the coordinated turn trajectory 
    expected_trajectory = coordinated_turn_trajectory(x0, dt, N, omega)

    assert np.allclose(model_trajectory, expected_trajectory, atol=1e-10, rtol=1e-10), "Trajectories do not match"


def test_F_matches_f():
    """
    Verify that the state transition matrix produces the same result as the nonlinear state propagation function.
    """

    omega = 2.5
    dt = 0.2

    motion = CoordinatedTurn2D(omega=omega, sigma_a=0)

    x = np.array([10.0, -5.0, 8.0, 3.0])

    expected = motion.f(x, dt)
    actual = motion.F(x, dt) @ x

    assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)

def test_omega_zero_raises():
    """
    Verify that zero turn rate raises ValueError
    """

    with pytest.raises(ValueError):
        CoordinatedTurn2D(omega=0, sigma_a=1.0)

def test_Q_symmetric_and_psd():
    """
    Verify that the process noise covariance is symmetric and positive semidefinite for several time intervals
    """

    motion = CoordinatedTurn2D(omega=2.0, sigma_a=1.0)

    dt_values = [0.01, 0.1, 0.5, 1.0, 2.0]

    for dt in dt_values:
        Q = motion.Q(dt)
        eigenvalues = np.linalg.eigvalsh(Q)

        # Check symmetry
        assert np.allclose(Q, Q.T, atol=1e-12)
        # Check positive semidefiniteness
        assert np.all(eigenvalues >= -1e-10), f"Q is not positive semidefinite for dt={dt}"


# To run: pytest tests/test_coordinated_turn.py -v
