import numpy as np
from fusion.filters.imm import InteractingMultipleModel
from fusion.filters.utils import gaussian_likelihood, gaussian_mixture_moments
from tracking.filters.kalman import KalmanFilter
from fusion.models.motion import CoordinatedTurn2D
from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D
from tracking.sim.trajectories import coordinated_turn_trajectory


def make_imm() -> InteractingMultipleModel:
    """
    Helper function to xreate a three-mode IMM with CV and opposing CT models
    """

    x0 = np.array([0.0, 0.0, 8.0, 2.0])
    P0 = np.diag([10.0, 10.0, 4.0, 4.0])

    measurement_model = LinearPosition2D(sigma_pos=1.5)

    filters = [
        KalmanFilter(
            ConstantVelocity2D(sigma_a=0.5),
            measurement_model,
            x0=x0.copy(),
            P0=P0.copy()
        ),
        KalmanFilter(
            CoordinatedTurn2D(omega=0.25, sigma_a=0.3),
            measurement_model,
            x0=x0.copy(),
            P0=P0.copy()
        ),
        KalmanFilter(
            CoordinatedTurn2D(omega=-0.25, sigma_a=0.3),
            measurement_model,
            x0=x0.copy(),
            P0=P0.copy()
        )
    ]

    # Rows represent previous modes; columns represent destination modes.
    transition_matrix = np.array([
        [0.96, 0.02, 0.02],
        [0.02, 0.96, 0.02],
        [0.02, 0.02, 0.96]
    ])

    mode_probs0 = np.array([1/3, 1/3, 1/3])

    return InteractingMultipleModel(filters, transition_matrix, mode_probs0)

def run_imm_on_trajectory(trajectory, seed) -> np.ndarray:
    """Helper function to run an IMM using noisy position measurements from a trajectory
    In: trajectory (N, 4), Ground-truth states;
        seed, Random seed for reproducible measurement noise;
   Out:
        Final posterior mode probabilities (3,)
    """

    rng = np.random.default_rng(seed)

    imm = make_imm()

    sigma_pos = 1.5
    dt = 0.1

    for x_true in trajectory:

        # Generate noisy position measurement
        z = x_true[:2] + rng.normal(0.0, sigma_pos, size=2)

        imm.predict(dt)
        imm.update(z)

    return imm.mode_probs.copy()







def test_mode_probs_sum_to_one():
    """
    Verify that mode probabilities remain normalized throughout repeated prediction and update cycles.
    """

    rng = np.random.default_rng(42)

    imm = make_imm()

    dt = 0.1
    n_steps = 50

    for _ in range(n_steps):

        # Predict
        imm.predict(dt)

        # Update using a noisy position measurement
        z = rng.normal(0, 5, size=2)
        imm.update(z)

        # Posterior mode probabilities must sum to one
        assert np.allclose(np.sum(imm.mode_probs), 1.0, atol=1e-12), "Posterior mode probabilities do not sum to one"


def test_mixture_moments_include_spread_term():
    """
    Verify that Gaussian mixture covariance includes the spread between individual state estimates
    """

    x1 = np.array([0.0, 0.0, 0.0, 0.0])
    x2 = np.array([10.0, 0.0, 0.0, 0.0])

    P = np.eye(4)

    states = [(x1, P), (x2, P)]

    weights = np.array([0.5, 0.5])

    x_comb, P_comb = gaussian_mixture_moments(states, weights)

    expected_x = np.array([5.0, 0.0, 0.0, 0.0])

    expected_P = np.diag([26.0, 1.0, 1.0, 1.0])

    assert np.allclose(x_comb, expected_x), "x_comb not equal to expected output"
    assert np.allclose(P_comb, expected_P), "P_comb not equal to expected output"




def test_straight_line_favors_cv():
    """
    Verify that the IMM favors the CV mode for straight-line motion
    """

    dt = 0.1
    n_steps = 120

    x0 = np.array([0.0, 0.0, 8.0, 2.0])

    # Generate deterministic constant-velocity ground truth
    true_motion = ConstantVelocity2D(sigma_a=0.0)

    trajectory = []
    x = x0.copy()

    for _ in range(n_steps):
        x = true_motion.f(x, dt)
        trajectory.append(x.copy())

    trajectory = np.asarray(trajectory)

    # Average final probabilities over multiple seeds
    seeds = [42, 123, 456, 789, 2026]

    final_probs = np.array([run_imm_on_trajectory(trajectory, seed) for seed in seeds])

    mean_probs = np.mean(final_probs, axis=0)

    # CV should have greater probability than either CT mode
    assert mean_probs[0] > mean_probs[1], "CV does not have greater prob than CT (CW)"
    assert mean_probs[0] > mean_probs[2], "CV does not have greater prob than CT (CCW)"

def test_turning_trajectory_favors_matching_ct_mode():
    """
    Verify that the IMM favors the CT mode matching the ground-truth turn direction
    """

    dt = 0.1
    n_steps = 120

    omega_true = 0.25

    x0 = np.array([0.0, 0.0, 8.0, 2.0])

    # Generate ground-truth coordinated-turn trajectory
    trajectory = coordinated_turn_trajectory(x0, dt, n_steps, omega_true)

    # Average final probabilities over multiple noise realizations
    seeds = [42, 123, 456, 789, 2026]

    final_probs = np.array([run_imm_on_trajectory(trajectory, seed) for seed in seeds])

    mean_probs = np.mean(final_probs, axis=0)

    # Positive CT mode should outperform CV and negative CT
    assert mean_probs[1] > mean_probs[0], "CT (CW) does not have greater prob than CV"
    assert mean_probs[1] > mean_probs[2], "CT (CW) does not have greater prob than CT (CCW)"


def test_degenerate_mode_probs_matches_single_filter():
    """
    Verify that a Gaussian mixture with all weight on one mode eturns exactly that mode's state and covariance
    """

    imm = make_imm()

    # Give all probability to the second mode
    imm.mode_probs = np.array([0.0, 1.0, 0.0])

    # Collect current filter estimates
    states = [(f.x, f.P) for f in imm.filters]

    # Combine using the degenerate probability vector
    x_comb, P_comb = gaussian_mixture_moments(states, imm.mode_probs)

    # The result must equal the second filter exactly
    assert np.array_equal(x_comb, imm.filters[1].x)
    assert np.array_equal(P_comb, imm.filters[1].P)

def test_predicted_mode_probs_sum_to_one():
    """
    Verify that predicted mode probabilities remain normalized
    """

    imm = make_imm()

    imm.predict(dt=0.1)

    assert np.allclose(np.sum(imm.predicted_mode_probs), 1.0, atol=1e-12)

# To run: pytest tests/test_imm.py -v