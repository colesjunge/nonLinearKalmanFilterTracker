import numpy as np
from tracking.models.measurement import LinearPosition2D
from fusion.models.adaptive_position import AdaptivePosition2D
from tracking.filters.kalman import KalmanFilter
from tracking.models.motion import ConstantVelocity2D

def test_same_state_as_linear():
    """
    Test that AdaptivePosition2D matches LinearPosition2D given no R updates
    """

    linear = LinearPosition2D(sigma_pos=1.0)
    adaptive = AdaptivePosition2D(R_init=np.eye(2))

    x = np.array([3.0, 4.0, 1.0, -2.0])

    assert np.allclose(linear.h(x), adaptive.h(x)), "Predicted measurements do not match"
    assert np.allclose(linear.H(x), adaptive.H(x)), "Measurement Jacobians do not match"

def test_r_returns_initial():
    """
    Test that AdaptivePosition2D stores initial R before updates
    """

    R_init = np.array([[4, 5], [8, 7]])

    model = AdaptivePosition2D(R_init=R_init)

    assert np.allclose(model.R(), R_init), "Initial R not stored"

def test_set_r_updates():
    """
    Test that AdaptivePosition2D updates initial R
    """

    R_init = np.array([[4, 5], [8, 7]])

    model = AdaptivePosition2D(R_init=R_init)

    R_update = np.array([[6, 4], [8, 10]])

    model.set_R(R=R_update)

    assert np.allclose(model.R(), R_update), "R not updated"

def test_large_r_distrusts_measurement():
    """
    Test that with large R, filter stays close to pre-update prediction
    """

    dt = 0.1
    x0 = np.array([95, 40, 5, 10])
    P0 = np.array([[10, 0, 0, 0], 
                    [0, 12, 0, 0], 
                    [0, 0, 34, 0], 
                    [0, 0, 0, 35]])

    z = np.array([0, 3]) # Deliberately far
    
    motion = ConstantVelocity2D(sigma_a=0.0)
    
    model = AdaptivePosition2D(R_init=np.eye(2))

    kf = KalmanFilter(motion=motion, measurement=model, x0=x0, P0=P0)
    kf.predict(dt)
    pre_update_pos = kf.x[:2]
    model.set_R(R= 1e6 * np.eye(2))

    kf.update(z)

    assert np.allclose(kf.x[:2], pre_update_pos, atol=1.0), "Filter not close to pre-update prediction"



def test_small_r_trusts_measurement():
    """
    Test that with small R, filter stays close to measurement
    """

    dt = 0.1
    x0 = np.array([95, 40, 5, 10])
    P0 = np.array([[10, 0, 0, 0], 
                    [0, 12, 0, 0], 
                    [0, 0, 34, 0], 
                    [0, 0, 0, 35]])

    z = np.array([90, 50]) # Deliberately close
    
    motion = ConstantVelocity2D(sigma_a=0.0)
    
    model = AdaptivePosition2D(R_init=np.eye(2))

    kf = KalmanFilter(motion=motion, measurement=model, x0=x0, P0=P0)
    kf.predict(dt)
    model.set_R(R= 1e-6 * np.eye(2))

    kf.update(z)

    assert np.allclose(kf.x[:2], z[:2], atol=1.0), "Filter not close to measurement"

# To run: pytest tests/test_adaptive_position.py -v

