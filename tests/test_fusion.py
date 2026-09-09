import numpy as np
from fusion.fuse.sequential import fuse_sequential
from fusion.filters.ekf import ExtendedKalmanFilter
from fusion.filters.ukf import UnscentedKalmanFilter
from fusion.models.radar import RadarRangeBearing
from fusion.models.eoir import EOIRBearingOnly
from tracking.models.motion import ConstantVelocity2D


def test_sequential_fusion_order_invariant():
    """
    Test that sequential fusion of measurements is order-invariant.
    """

    # Filter setup
    eoir_model = EOIRBearingOnly(sigma_bearing=0.1)
    radar_model = RadarRangeBearing(sigma_range=20.0, sigma_bearing=0.1)
    motion_model = ConstantVelocity2D(sigma_a=0.0)


    x0_filter = np.array([95, 40, 5, 10]) # Initial state uncertainty (random numbers I've chosen)
    P0 = np.array([[10, 0, 0, 0], 
                    [0, 12, 0, 0], 
                    [0, 0, 34, 0], 
                    [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)

    ekf_1 = ExtendedKalmanFilter(motion_model, eoir_model, x0=x0_filter, P0=P0)
    ekf_2 = ExtendedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)
    ukf_1 = UnscentedKalmanFilter(motion_model, eoir_model, x0=x0_filter, P0=P0)
    ukf_2 = UnscentedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)

    # Define two measurement vectors
    eoir_measurement = np.array([0.5])  # Example bearing measurement
    radar_measurement = np.array([100.0, 0.5])  # Example range and bearing measurement

    # Fuse measurements in one order
    fuse_sequential(ekf_1, [(eoir_model, eoir_measurement), (radar_model, radar_measurement)])
    fuse_sequential(ukf_1, [(eoir_model, eoir_measurement), (radar_model, radar_measurement)])

    # Fuse measurements in the reverse order
    fuse_sequential(ekf_2, [(radar_model, radar_measurement), (eoir_model, eoir_measurement)])
    fuse_sequential(ukf_2, [(radar_model, radar_measurement), (eoir_model, eoir_measurement)])

    # Check if the updated measurements are the same regardless of the order (using rtol to account for numerical differences)
    assert np.allclose(ekf_1.x, ekf_2.x, rtol=0.1), "EKF State estimates differ based on the order of fusion."
    assert np.allclose(ekf_1.P, ekf_2.P, rtol=0.1), "EKF Covariance estimates differ based on the order of fusion."

    assert np.allclose(ukf_1.x, ukf_2.x, rtol=0.1), "UKF State estimates differ based on the order of fusion."
    assert np.allclose(ukf_1.P, ukf_2.P, rtol=0.1), "UKF Covariance estimates differ based on the order of fusion."

def test_fusion_reduces_covariance():
    """
    Test that sequential fusion of measurements reduces the covariance of the filter.
    """

    # Filter setup
    eoir_model = EOIRBearingOnly(sigma_bearing=0.1)
    radar_model = RadarRangeBearing(sigma_range=20.0, sigma_bearing=0.1)
    motion_model = ConstantVelocity2D(sigma_a=0.0)


    x0_filter = np.array([95, 40, 5, 10]) # Initial state uncertainty (random numbers I've chosen)
    P0 = np.array([[10, 0, 0, 0], 
                    [0, 12, 0, 0], 
                    [0, 0, 34, 0], 
                    [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)

    ekf_eoir = ExtendedKalmanFilter(motion_model, eoir_model, x0=x0_filter, P0=P0)
    ekf_radar = ExtendedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)
    ekf_fused = ExtendedKalmanFilter(motion_model, eoir_model, x0=x0_filter, P0=P0)

    ukf_radar = UnscentedKalmanFilter(motion_model, radar_model, x0=x0_filter, P0=P0)
    ukf_eoir = UnscentedKalmanFilter(motion_model, eoir_model, x0=x0_filter, P0=P0)
    ukf_fused = UnscentedKalmanFilter(motion_model, eoir_model, x0=x0_filter, P0=P0)

    eoir_measurement = np.array([0.5])  # Example bearing measurement
    radar_measurement = np.array([100.0, 0.5])  # Example range and bearing measurement

    # Fuse measurements sequentially
    fuse_sequential(ekf_fused, [(eoir_model, eoir_measurement), (radar_model, radar_measurement)])
    fuse_sequential(ukf_fused, [(eoir_model, eoir_measurement), (radar_model, radar_measurement)])

    # Update non fused filters with individual measurements
    ekf_eoir.update(eoir_measurement)
    ekf_radar.update(radar_measurement)

    ukf_eoir.update(eoir_measurement)
    ukf_radar.update(radar_measurement)

    # Check if the covariance matrix of the fused filter is less than or equal to the individual filters
    assert np.linalg.det(ekf_fused.P) < np.linalg.det(ekf_radar.P), "Fused EKF covariance is not less than or equal to radar EKF covariance."
    assert np.linalg.det(ekf_fused.P) < np.linalg.det(ekf_eoir.P), "Fused EKF covariance is not less than or equal to EOIR EKF covariance."

    assert np.linalg.det(ukf_fused.P) < np.linalg.det(ukf_radar.P), "Fused UKF covariance is not less than or equal to radar UKF covariance."
    assert np.linalg.det(ukf_fused.P) < np.linalg.det(ukf_eoir.P), "Fused UKF covariance is not less than or equal to EOIR UKF covariance."

# To run: pytest tests/test_fusion.py -v