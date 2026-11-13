import numpy as np

from tracking.sim.trajectories import coordinated_turn_trajectory, constant_velocity_trajectory


def waypoint_trajectory(x0: np.ndarray,
                        dt: float,
                        n_steps: int,
                        rng: np.random.Generator,
                        turn_prob: float,
                        leg_duration_range: tuple[float, float],
                        omega_range: tuple[float, float]) -> np.ndarray:
    """ Creates a random sequence of continuous straight and turn (using coordinated_turn_trajectory) trajectories
    In: x0, (4,) initial [px, py, vx, vy];
        dt, seconds per step;
        n_steps, total length of the returned trajectory
        rng, drives every leg decision (type, duration, omega magnitude, direction);
        turn_prob, probability that leg is a turn rather than straight;
        leg_duration_range, (min_seconds, max_seconds) each leg's duration drawn uniformly;
        omega_range: (min_magnitude, max_magnitude) in rad/s for turn legs (floor of .05 rad/s and direction seperate);
    Out: trajectory (n_steps, 4) concatenated ground-truth states across all legs
    """

    trajectories = []
    current_x = x0.copy()
    current_step = 0

    min_dur = leg_duration_range[0]
    max_dur = leg_duration_range[1]
    # To prevent instability set min to 0.05
    min_omega = max(omega_range[0], 0.05)
    max_omega = omega_range[1]


    while current_step < n_steps:

        # Determine if leg is turn
        is_turn = rng.random() < turn_prob

        # Determine leg duration and convert to amount of steps
        leg_duration = rng.uniform(min_dur, max_dur)
        n_steps_leg = max(round(leg_duration / dt), 5) # Ensure at least 5 steps per leg

        # If turn, need to find omega and CW/CCW
        if is_turn:
            leg_omega = rng.choice([-1, 1]) * rng.uniform(min_omega, max_omega)
            trajectories.append(coordinated_turn_trajectory(current_x, dt, n_steps_leg, leg_omega))
        else:
            trajectories.append(constant_velocity_trajectory(current_x, dt, n_steps_leg))

        current_x = trajectories[-1][-1].copy()
        current_step += n_steps_leg if len(trajectories) == 1 else n_steps_leg - 1

    # Remove duplicate initial states from all legs after the first
    trajectories = [trajectories[0]] + [leg[1:] for leg in trajectories[1:]]

    # Concatenate all legs and trim to n_steps
    trajectory = np.concatenate(trajectories, axis=0)[:n_steps]

    return trajectory





        










