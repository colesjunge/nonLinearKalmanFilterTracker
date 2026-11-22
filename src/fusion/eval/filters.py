import numpy as np

from tracking.filters.base import BayesFilter
from tracking.models.motion import ConstantVelocity2D
from fusion.models.radar import RadarRangeBearing
from fusion.models.eoir import EOIRBearingOnly
from fusion.models.motion import CoordinatedTurn2D
from fusion.filters.imm import InteractingMultipleModel
from fusion.filters.ekf import ExtendedKalmanFilter


def build_benign_filter(sigma_a: float, trajectory: np.ndarray, z_seq: np.ndarray, filter_cls: type[BayesFilter], config: dict) -> BayesFilter:
    """ Builds benign filter
    In: sigma_a, the value being swept/root-found this call;
        trajectory (n_steps-1, 4), sequence of states (unused in this builder);
        z_seq (n_steps-1, 2), sequence of raw noisy range-bearing measurement sequences;
        filter_cls, filter builder (UKF or EKF);
        config, final_comparison.yaml
    Out: benign_filt, benign filter
    
    """

    px, py = z_seq[0][0]*np.cos(z_seq[0][1]), z_seq[0][0]*np.sin(z_seq[0][1])
    x0 = np.array([px, py, 0, 0])
    P0 = np.array(config['initialization']['P0'])

    sigma_range = config["scenarios"]["benign"]["sigma_range"]
    sigma_bearing = config["scenarios"]["benign"]["sigma_bearing"]

    benign_filt = filter_cls(ConstantVelocity2D(sigma_a), RadarRangeBearing(sigma_range, sigma_bearing), x0, P0)
   
    return benign_filt

def build_maneuvering_filter(sigma_a: float, trajectory: np.ndarray, z_seq: np.ndarray, filter_cls: type[BayesFilter], config: dict) -> BayesFilter:
    """ Builds maneuvering filter (This will also be used for the CV mode in the IMM)
    In: sigma_a, the value being swept/root-found this call;
        trajectory (n_steps-1, 4), sequence of states (unused in this builder);
        z_seq (n_steps-1, 2), sequence of raw noisy range-bearing measurement sequences;
        filter_cls, filter builder (UKF or EKF);
        config, final_comparison.yaml
    Out: maneuvering_filt, maneuvering filter
    
    """
    px, py = z_seq[0][0]*np.cos(z_seq[0][1]), z_seq[0][0]*np.sin(z_seq[0][1])
    x0 = np.array([px, py, 0, 0])
    P0 = np.array(config['initialization']['P0'])

    sigma_range = config["scenarios"]["maneuvering"]["sigma_range"]
    sigma_bearing = config["scenarios"]["maneuvering"]["sigma_bearing"]

    maneuvering_filt = filter_cls(ConstantVelocity2D(sigma_a), RadarRangeBearing(sigma_range, sigma_bearing), x0, P0)
   
    return maneuvering_filt

def build_ct_mode_filter(sigma_a: float, trajectory: np.ndarray, z_seq: np.ndarray, filter_cls: type[BayesFilter], omega: float, config: dict) -> BayesFilter:
    """ Builds CT mode filter for IMM
    In: sigma_a, the value being swept/root-found this call;
        trajectory (n_steps-1, 4), sequence of states (unused in this builder);
        z_seq (n_steps-1, 2), sequence of raw noisy range-bearing measurement sequences;
        filter_cls, filter builder (EKF)
        omega, found from compute_omega;
        config, final_comparison.yaml
    Out: ct_mode_filt, CT mode filter
    
    """
    px, py = z_seq[0][0]*np.cos(z_seq[0][1]), z_seq[0][0]*np.sin(z_seq[0][1])
    x0 = np.array([px, py, 0, 0])
    P0 = np.array(config['initialization']['P0'])

    sigma_range = config["scenarios"]["maneuvering"]["sigma_range"]
    sigma_bearing = config["scenarios"]["maneuvering"]["sigma_bearing"]

    ct_mode_filt = filter_cls(CoordinatedTurn2D(omega, sigma_a), RadarRangeBearing(sigma_range, sigma_bearing), x0, P0)
   
    return ct_mode_filt

def build_contaminated_filter(sigma_a: float, trajectory: np.ndarray, z_seq: np.ndarray, filter_cls: type[BayesFilter], config: dict) -> BayesFilter:
    """ Builds contaminated filter
    In: sigma_a, the value being swept/root-found this call;
        trajectory (n_steps-1, 4), sequence of states (unused in this builder);
        z_seq (n_steps-1, 2), sequence of raw noisy range-bearing measurement sequences;
        filter_cls, filter builder (UKF or EKF);
        config, final_comparison.yaml
    Out: contaminated_filt, contaminated filter
    
    """

    # To reduce the contamination risk on z_seq which can warp the nees estimates use median of first 5 steps
    # In first five steps, window is short enough that true bearing doesn't approach wrap around so median works
    range_est = np.median(z_seq[:5, 0])
    bearing_est = np.median(z_seq[:5, 1])
    
    px, py = range_est*np.cos(bearing_est), range_est*np.sin(bearing_est)
    x0 = np.array([px, py, 0, 0])
    P0 = np.array(config['initialization']['P0'])

    sigma_range = config["scenarios"]["contaminated"]["sigma_range"]
    sigma_bearing = config["scenarios"]["contaminated"]["sigma_bearing"]

    contaminated_filt = filter_cls(ConstantVelocity2D(sigma_a), RadarRangeBearing(sigma_range, sigma_bearing), x0, P0)
   
    return contaminated_filt

def build_bearing_only_filter(sigma_a: float, trajectory: np.ndarray, z_seq: np.ndarray, filter_cls: type[BayesFilter], config: dict) -> BayesFilter:
    """ Builds bearing only filter
    In: sigma_a, the value being swept/root-found this call;
        trajectory (n_steps-1, 4), sequence of states (unused in this builder);
        z_seq (n_steps-1, 1), sequence of raw noisy bearing only measurement sequences;
        filter_cls, filter builder (UKF or EKF);
        config, final_comparison.yaml
    Out: bearing_only_filt, bearing only filter
    
    """

    # To reduce the contamination risk on z_seq which can warp the nees estimates use median of first 5 steps
    # In first five steps, window is short enough that true bearing doesn't approach wrap around so median works
    bearing_est = np.median(z_seq[:5, 0])
    
    default_range_guess = config["scenarios"]["bearing_only"]["default_range_guess"]
    sensor_pos = config["scenarios"]["bearing_only"]["sensor_pos"]

    px = sensor_pos[0] + default_range_guess*np.cos(bearing_est)
    py = sensor_pos[1] + default_range_guess*np.sin(bearing_est)
    x0 = np.array([px, py, 0, 0])
    P0 = np.array(config['initialization']['P0_bearing'])

    sigma_bearing = config["scenarios"]["bearing_only"]["sigma_bearing"]


    bearing_only_filt = filter_cls(ConstantVelocity2D(sigma_a), EOIRBearingOnly(sigma_bearing, np.array(sensor_pos)), x0, P0)
   
    return bearing_only_filt

def build_dual_sensor_sequential_filter(sigma_a: float, trajectory: np.ndarray, z_radar_seq: np.ndarray, z_eoir_seq: np.ndarray, filter_cls: type[BayesFilter], config: dict) -> BayesFilter:
    """ Builds dual sensor filter with sequential fusing
    In: sigma_a, the value being swept/root-found this call;
        trajectory (n_steps-1, 4), sequence of states (unused in this builder);
        z_radar_seq (n_steps-1, 2), sequence of raw noisy range-bearing measurement sequences;
        z_eoir_seq (n_steps-1, 1), sequence of raw noisy bearing only measurement sequences;
        filter_cls, filter builder (UKF or EKF);
        config, final_comparison.yaml
    Out: dual_sensor_sequential_filt, dual sensor sequential filter
    """

    radar_pos = config["scenarios"]["dual_sensor_fusion"]["radar_pos"]

    px = radar_pos[0] + z_radar_seq[0][0]*np.cos(z_radar_seq[0][1])
    py = radar_pos[1] + z_radar_seq[0][0]*np.sin(z_radar_seq[0][1])
    x0 = np.array([px, py, 0, 0])
    P0 = np.array(config['initialization']['P0'])

    sigma_range = config["scenarios"]["dual_sensor_fusion"]["sigma_range"]
    sigma_bearing = config["scenarios"]["dual_sensor_fusion"]["sigma_bearing"]

    dual_sensor_sequential_filt = filter_cls(ConstantVelocity2D(sigma_a), RadarRangeBearing(sigma_range, sigma_bearing,np.array(radar_pos)), x0, P0)
   
    return dual_sensor_sequential_filt

def build_dual_sensor_track_pair(sigma_a: float, trajectory: np.ndarray, z_radar_seq: np.ndarray, z_eoir_seq: np.ndarray, filter_cls: type[BayesFilter], config: dict) -> tuple[BayesFilter, BayesFilter]:
    """ Builds dual sensor filters (one RadarRangeBearing, one EOIRBearingOnly) with for track level fusion
    In: sigma_a, the value being swept/root-found this call;
        trajectory (n_steps-1, 4), sequence of states (unused in this builder);
        z_radar_seq (n_steps-1, 2), sequence of raw noisy range-bearing measurement sequences;
        z_eoir_seq (n_steps-1, 1), sequence of raw noisy bearing only measurement sequences;
        filter_cls, filter builder (UKF or EKF);
        config, final_comparison.yaml
    Out: dual_sensor_track_pair_filts, (RadarRangeBearing filter, EOIRBearingOnly filter)
    """
    radar_pos = config["scenarios"]["dual_sensor_fusion"]["radar_pos"]
    eoir_pos = config["scenarios"]["dual_sensor_fusion"]["eoir_pos"]

    px = radar_pos[0] + z_radar_seq[0][0]*np.cos(z_radar_seq[0][1])
    py = radar_pos[1] + z_radar_seq[0][0]*np.sin(z_radar_seq[0][1])
    x0 = np.array([px, py, 0, 0])
    P0 = np.array(config['initialization']['P0'])

    sigma_range = config["scenarios"]["dual_sensor_fusion"]["sigma_range"]
    sigma_bearing = config["scenarios"]["dual_sensor_fusion"]["sigma_bearing"]


    dual_sensor_radar_filt = filter_cls(ConstantVelocity2D(sigma_a), RadarRangeBearing(sigma_range, sigma_bearing,np.array(radar_pos)), x0, P0)
    dual_sensor_eoir_filt = filter_cls(ConstantVelocity2D(sigma_a), EOIRBearingOnly(sigma_bearing, np.array(eoir_pos)), x0, P0)

    dual_sensor_track_pair_filts = (dual_sensor_radar_filt, dual_sensor_eoir_filt)
    
    return dual_sensor_track_pair_filts

def build_imm_filter(trajectory: np.ndarray, z_seq: np.ndarray, config: dict) -> InteractingMultipleModel:
    """ Builds IMM filter (CV, CT+, CT-)
    In: sigma_a, the value being swept/root-found this call;
        trajectory (n_steps-1, 4), sequence of states (unused in this builder);
        z_seq (n_steps-1, 2), sequence of raw noisy range-bearing measurement sequences;
        config, final_comparison.yaml
    Out: imm_filt, IMM filter
    
    """
    px, py = z_seq[0][0]*np.cos(z_seq[0][1]), z_seq[0][0]*np.sin(z_seq[0][1])
    x0 = np.array([px, py, 0, 0])
    P0 = np.array(config['initialization']['P0'])

    sigma_range = config["scenarios"]["maneuvering"]["sigma_range"]
    sigma_bearing = config["scenarios"]["maneuvering"]["sigma_bearing"]

    omega = config["tuned"]["maneuvering"]["omega"]
    sigma_a_cv = config["tuned"]["maneuvering"]["modes"]["cv"]
    sigma_a_ct_plus = config["tuned"]["maneuvering"]["modes"]["ct_plus"]
    sigma_a_ct_minus = config["tuned"]["maneuvering"]["modes"]["ct_minus"]

    transition_matrix = np.array(config["scenarios"]["maneuvering"]["transition_matrix"])
    mode_probs0 = np.array(config["scenarios"]["maneuvering"]["mode_probs0"])


    cv = ExtendedKalmanFilter(ConstantVelocity2D(sigma_a_cv), RadarRangeBearing(sigma_range, sigma_bearing), x0, P0)
    ct_plus = ExtendedKalmanFilter(CoordinatedTurn2D(omega, sigma_a_ct_plus), RadarRangeBearing(sigma_range, sigma_bearing), x0, P0)
    ct_minus = ExtendedKalmanFilter(CoordinatedTurn2D(-omega, sigma_a_ct_minus), RadarRangeBearing(sigma_range, sigma_bearing), x0, P0)

    imm_filt = InteractingMultipleModel([cv, ct_plus, ct_minus], transition_matrix, mode_probs0)

    return imm_filt