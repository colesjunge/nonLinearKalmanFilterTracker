import numpy as np
from tracking.sim.trajectories import constant_velocity_trajectory
from fusion.models.radar import RadarRangeBearing
from fusion.sim.maneuvers import waypoint_trajectory
from fusion.sim.noise import contaminated_gaussian_noise
from fusion.models.eoir import EOIRBearingOnly


def build_benign_scenario(rng: np.random.Generator, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """ Builds a benign scenario (random constant-velocity trajectory, clean Gaussian-noise radar range-bearing measurements)
    In: rng, for x0 sampling (position/speed/heading) and measurement noise;
        config, dt/n_steps and config['scenarios']['benign']['sigma_range']/['sigma_bearing']
    Out: trajectory (n_steps-1, 4) ground-truth states (1 to n_steps-1);
         z_sequence (n_steps-1, 2) noisy range-bearing measurements (1 to n_steps-1)
    """
    
    dt = config["dt"]
    n_steps = config["n_steps"]

    sigma_range = config["scenarios"]["benign"]["sigma_range"]
    sigma_bearing= config["scenarios"]["benign"]["sigma_bearing"]

    radar_model = RadarRangeBearing(sigma_range, sigma_bearing)

    # Random start state
    x = rng.uniform(-100, 100)
    y = rng.uniform(-100, 100)

    speed = rng.uniform(5, 30)
    heading = rng.uniform(0, 2 * np.pi)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    x0_true = np.array([x, y, vx, vy])

    # Ground truth trajectory
    trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

    z_sequence = []

    for i in range(1, n_steps):
        z_sequence.append(radar_model.h(trajectory[i]) + rng.multivariate_normal(np.zeros(2), radar_model.R()))

    z_sequence = np.array(z_sequence)

    # Ignore first trajectory index to align with z_sequence which begins at step 1
    return trajectory[1:], z_sequence


def build_maneuvering_scenario(rng: np.random.Generator, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """ Builds a maneuvering scenario (random waypoint trajectory, clean Gaussian-noise radar range-bearing measurements)
    In: rng, for x0 sampling (position/speed/heading) and measurement noise;
        config, dt/n_steps and config['scenarios']['maneuvering']['sigma_range']/['sigma_bearing']/['turn_prob']/
        [leg_duration_range']/['omega_range']
    Out: trajectory (n_steps-1, 4) ground-truth states (1 to n_steps-1);
         z_sequence (n_steps-1, 2) noisy range-bearing measurements (1 to n_steps-1)
    """
    
    dt = config["dt"]
    n_steps = config["n_steps"]

    sigma_range = config["scenarios"]["maneuvering"]["sigma_range"]
    sigma_bearing= config["scenarios"]["maneuvering"]["sigma_bearing"]
    turn_prob = config["scenarios"]["maneuvering"]["turn_prob"]
    leg_duration_range = config["scenarios"]["maneuvering"]["leg_duration_range"]
    omega_range = config["scenarios"]["maneuvering"]["omega_range"]

    radar_model = RadarRangeBearing(sigma_range, sigma_bearing)

    # Random start state
    x = rng.uniform(-100, 100)
    y = rng.uniform(-100, 100)

    speed = rng.uniform(5, 30)
    heading = rng.uniform(0, 2 * np.pi)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    x0_true = np.array([x, y, vx, vy])

    # Ground truth trajectory
    trajectory = waypoint_trajectory(x0_true, dt, n_steps, rng, turn_prob, leg_duration_range, omega_range)

    z_sequence = []

    for i in range(1, n_steps):
        z_sequence.append(radar_model.h(trajectory[i]) + rng.multivariate_normal(np.zeros(2), radar_model.R()))

    z_sequence = np.array(z_sequence)

    # Ignore first trajectory index to align with z_sequence which begins at step 1
    return trajectory[1:], z_sequence



def build_contaminated_scenario(rng: np.random.Generator, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """ Builds a contaminated scenario (random constant-velocity trajectory, contaminated-noise radar range-bearing measurements)
    In: rng, for x0 sampling (position/speed/heading) and measurement noise;
        config, dt/n_steps and config['scenarios']['contaminated']['sigma_range']/['sigma_bearing']/['epsilon']/['kappa']
    Out: trajectory (n_steps-1, 4) ground-truth states (1 to n_steps-1);
         z_sequence (n_steps-1, 2) noisy range-bearing measurements (1 to n_steps-1)
    """
    
    dt = config["dt"]
    n_steps = config["n_steps"]

    sigma_range = config["scenarios"]["contaminated"]["sigma_range"]
    sigma_bearing= config["scenarios"]["contaminated"]["sigma_bearing"]
    epsilon = config["scenarios"]["contaminated"]["epsilon"]
    kappa = config["scenarios"]["contaminated"]["kappa"]


    radar_model = RadarRangeBearing(sigma_range, sigma_bearing)

    # Random start state
    x = rng.uniform(-100, 100)
    y = rng.uniform(-100, 100)

    speed = rng.uniform(5, 30)
    heading = rng.uniform(0, 2 * np.pi)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    x0_true = np.array([x, y, vx, vy])

    # Ground truth trajectory
    trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

    z_sequence = []

    for i in range(1, n_steps):
        z_sequence.append(radar_model.h(trajectory[i]) + contaminated_gaussian_noise(rng, radar_model.R(), epsilon, kappa))

    z_sequence = np.array(z_sequence)

    # Ignore first trajectory index to align with z_sequence which begins at step 1
    return trajectory[1:], z_sequence



def build_bearing_only_scenario(rng: np.random.Generator, config: dict) -> tuple[np.ndarray, np.ndarray]:
    """ Builds a bearing only scenario (random constant-velocity trajectory, contaminated-noise EOIR bearing-only measurements)
    In: rng, for x0 sampling (position/speed/heading) and measurement noise;
        config, dt/n_steps and config['scenarios']['bearing_only']['sensor_pos']/['sigma_bearing']/['epsilon']/['kappa']
    Out: trajectory (n_steps-1, 4) ground-truth states (1 to n_steps-1);
         z_sequence (n_steps-1, 1) noisy bearing-only measurements (1 to n_steps-1)
    """
    
    dt = config["dt"]
    n_steps = config["n_steps"]

    sensor_pos = config["scenarios"]["bearing_only"]["sensor_pos"]
    sigma_bearing= config["scenarios"]["bearing_only"]["sigma_bearing"]
    epsilon = config["scenarios"]["bearing_only"]["epsilon"]
    kappa = config["scenarios"]["bearing_only"]["kappa"]


    eoir_model = EOIRBearingOnly(sigma_bearing=sigma_bearing, sensor_pos=np.array(sensor_pos))

    # Random start state
    x = rng.uniform(-100, 100)
    y = rng.uniform(-100, 100)

    speed = rng.uniform(5, 30)
    heading = rng.uniform(0, 2 * np.pi)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    x0_true = np.array([x, y, vx, vy])

    # Ground truth trajectory
    trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

    z_sequence = []

    for i in range(1, n_steps):
        z_sequence.append(eoir_model.h(trajectory[i]) + contaminated_gaussian_noise(rng, eoir_model.R(), epsilon, kappa))

    z_sequence = np.array(z_sequence)

    # Ignore first trajectory index to align with z_sequence which begins at step 1
    return trajectory[1:], z_sequence



def build_dual_sensor_fusion_scenario(rng: np.random.Generator, config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Builds a dual_sensor scenario (random constant-velocity trajectory, clean Gaussian-noise radar range-bearing measurements and bearing_only measurements)
    In: rng, for x0 sampling (position/speed/heading) and measurement noise;
        config, dt/n_steps and config['scenarios']['dual_sensor_fusion']['sigma_range']/['sigma_bearing']/['radar_pos']/['eoir_pos']
        This does not perform the fusion itself
    Out: trajectory (n_steps-1, 4) ground-truth states (1 to n_steps-1);
        z_radar_sequence (n_steps-1, 2) noisy range-bearing measurements (1 to n_steps-1);
        z_radar_sequence (n_steps-1, 1) noisy bearing_only measurements (1 to n_steps-1)
    """
    
    dt = config["dt"]
    n_steps = config["n_steps"]

    sigma_range = config["scenarios"]["dual_sensor_fusion"]["sigma_range"]
    sigma_bearing = config["scenarios"]["dual_sensor_fusion"]["sigma_bearing"]
    radar_pos = config["scenarios"]["dual_sensor_fusion"]["radar_pos"]
    eoir_pos = config["scenarios"]["dual_sensor_fusion"]["eoir_pos"]

    radar_model = RadarRangeBearing(sigma_range, sigma_bearing, sensor_pos=np.array(radar_pos))
    eoir_model = EOIRBearingOnly(sigma_bearing=sigma_bearing, sensor_pos=np.array(eoir_pos))

    # Random start state
    x = rng.uniform(-100, 100)
    y = rng.uniform(-100, 100)

    speed = rng.uniform(5, 30)
    heading = rng.uniform(0, 2 * np.pi)

    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)

    x0_true = np.array([x, y, vx, vy])

    # Ground truth trajectory
    trajectory = constant_velocity_trajectory(x0_true, dt, n_steps)

    z_radar_sequence = []
    z_eoir_sequence = []

    for i in range(1, n_steps):
        z_radar_sequence.append(radar_model.h(trajectory[i]) + rng.multivariate_normal(np.zeros(2), radar_model.R()))
        z_eoir_sequence.append(eoir_model.h(trajectory[i]) + rng.multivariate_normal(np.zeros(1), eoir_model.R()))

    z_radar_sequence = np.array(z_radar_sequence)
    z_eoir_sequence = np.array(z_eoir_sequence)

    # Ignore first trajectory index to align with z_sequences which begins at step 1
    return trajectory[1:], z_radar_sequence, z_eoir_sequence


