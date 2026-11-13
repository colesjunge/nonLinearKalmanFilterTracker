import numpy as np
from fusion.sim.maneuvers import waypoint_trajectory

def test_waypoint_trajectory_output_shape():
    """
    Test output shape is (n_step, 4)
    """

    x0 = np.array([3, 5, 7, 3]) # random numbers I've chosen
    dt = .01
    n_steps = 100
    rng = np.random.default_rng(42)
    turn_prob = .4
    leg_duration_range = (.1, .5)
    omega_range = (0, 5)

    trajectory = waypoint_trajectory(x0=x0, dt=dt, n_steps=n_steps, rng=rng, turn_prob=turn_prob, leg_duration_range=leg_duration_range, omega_range=omega_range)

    assert trajectory.shape == (n_steps, 4), "Output shape not (n_steps, 4)"


def test_waypoint_trajectory_no_duplicates():
    """
    Test that no two consecutive rows in the output are exactly equal
    """
    x0 = np.array([3, 5, 7, 3]) # random numbers I've chosen
    dt = .01
    n_steps = 100
    rng = np.random.default_rng(42)
    turn_prob = .4
    leg_duration_range = (.1, .5)
    omega_range = (0, 5)

    trajectory = waypoint_trajectory(x0=x0, dt=dt, n_steps=n_steps, rng=rng, turn_prob=turn_prob, leg_duration_range=leg_duration_range, omega_range=omega_range)
    differences = np.diff(trajectory, axis=0)

    assert not np.any(np.all(differences == 0, axis=1)), "Consecutive rows in output are equal"

def test_waypoint_trajectory_unmodified_x0():
    """
    Test that original x0 remains unmodified after calling
    """
    x0_untouched = np.array([3, 5, 7, 3])
    x0 = np.array([3, 5, 7, 3]) # random numbers I've chosen
    dt = .01
    n_steps = 100
    rng = np.random.default_rng(42)
    turn_prob = .4
    leg_duration_range = (.1, .5)
    omega_range = (0, 5)

    trajectory = waypoint_trajectory(x0=x0, dt=dt, n_steps=n_steps, rng=rng, turn_prob=turn_prob, leg_duration_range=leg_duration_range, omega_range=omega_range)

    assert np.array_equal(x0_untouched, x0), "x0 was modified"

# To run: pytest tests/test_waypoint_trajectory.py -v

