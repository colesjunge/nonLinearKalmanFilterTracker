import numpy as np
from scipy import optimize
from tracking.filters.kalman import KalmanFilter
from fusion.filters.ekf import ExtendedKalmanFilter
from tracking.models.measurement import LinearPosition2D
from tracking.models.motion import ConstantVelocity2D
from fusion.models.radar import RadarRangeBearing


def test_ekf_matches_kf_on_linear_model():
    """
    Test that the Extended Kalman Filter behaves like a standard Kalman Filter (KF)
    Should output same x and P
    """

    dt = 0.1
    n_steps = 100
    x0_true = np.array([0, 0, 1, 1]) # px, py, vx, vy

    # Constant-velocity trajectory
    times = np.arange(n_steps) * dt

    trajectory = np.column_stack([
        x0_true[0] + x0_true[2] * times,
        x0_true[1] + x0_true[3] * times,
        np.full(n_steps, x0_true[2]),
        np.full(n_steps, x0_true[3]),
    ])

    # Filter setup
    motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
    measurement_model = LinearPosition2D(sigma_pos=1e-6) # Small measurement noise to avoid errors in the Kalman gain calculation

    x0_filter = np.array([.5, .3, 5, 10]) # Initial state uncertainty (random numbers I've chosen)
    P0 = np.array([[10, 0, 0, 0], 
                   [0, 12, 0, 0], 
                   [0, 0, 34, 0], 
                   [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)

    kf = KalmanFilter(motion_model, measurement_model, x0=x0_filter, P0=P0)
    ekf = ExtendedKalmanFilter(motion_model, measurement_model, x0=x0_filter, P0=P0)

    # Test KF vs EKF on linear model
    for i in range(1, n_steps):
        kf.predict(dt)
        kf.update(trajectory[i][:2])

        ekf.predict(dt)
        ekf.update(trajectory[i][:2])

        # Check that the state estimates and covariance matrices are close
        assert np.allclose(kf.x, ekf.x, atol=1e-10), f"KF and EKF state estimates differ at step {i}"
        assert np.allclose(kf.P, ekf.P, atol=1e-10), f"KF and EKF covariance estimates differ at step {i}"



def test_jacobian_matches_numerical():
    """
    Test that the Jacobian of the radar measurement model matches a numerical approximation.
    """

    x = np.array([1.0, 1.0, 0.5, -0.2])  # Random state vector
    radar_model = RadarRangeBearing(sigma_range=20.0, sigma_bearing=0.01)
    H = radar_model.H(x)
    
    H_num = optimize.approx_fprime(x, lambda x: radar_model.h(x))

    # Check that the analytical Jacobian matches the numerical approximation
    assert np.allclose(H, H_num, atol=1e-6), "Analytical Jacobian does not match numerical approximation"


def test_ekf_handles_target_at_origin():
    """
    Test that the Extended Kalman Filter can handle a target at the origin.
    """
    x = np.array([0.0, 0.0, 0.5, -0.2])  # Target at the origin
    radar_model = RadarRangeBearing(sigma_range=20.0, sigma_bearing=0.01)
    H = radar_model.H(x)

    # Check that the Jacobian does not contain NaN or infinite values
    assert np.all(np.isfinite(H)), "Jacobian contains infinite/NaN values"


# To run: pytest tests/test_ekf.py -v