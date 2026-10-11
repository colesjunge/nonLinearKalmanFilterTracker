import numpy as np
import torch
from tracking.filters.kalman import KalmanFilter
from tracking.models.measurement import LinearPosition2D
from tracking.models.motion import ConstantVelocity2D
from fusion.learned.diff_kf import DifferentiableKF

def test_diff_kf_matches_numpy_kf():
    """
    Test that diff_kf matches standard kf
    """

    rng = np.random.default_rng(42)

    dt = 0.1
    K = 10
    x0_true = np.array([0, 0, 1, 1]) # px, py, vx, vy
    

    # Constant-velocity trajectory
    times = np.arange(1, K+1) * dt

    trajectory = np.column_stack([
        x0_true[0] + x0_true[2] * times,
        x0_true[1] + x0_true[3] * times,
    ])

    sigma_pos = 20
    z = trajectory + rng.normal(0.0, sigma_pos, size=(K, 2))

    # Filter setup
    motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
    measurement_model = LinearPosition2D(sigma_pos=sigma_pos) 

    x0_filter = np.array([.5, .3, 5, 10]) # Initial state uncertainty (random numbers I've chosen)
    P0 = np.array([[10, 0, 0, 0], 
                [0, 12, 0, 0], 
                [0, 0, 34, 0], 
                [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)

    # Reference KF
    kf = KalmanFilter(motion_model, measurement_model, x0=x0_filter, P0=P0)

    x_ref = []
    P_ref = []

    for t in range(K):
        kf.predict(dt)
        kf.update(z[t])

        x_ref.append(kf.x.copy())
        P_ref.append(kf.P.copy())

    x_ref = np.stack(x_ref)
    P_ref = np.stack(P_ref)

    # Extract F, H, Q, R matrices
    F = motion_model.F(x0_filter, dt)
    H = measurement_model.H(x0_filter)
    Q = motion_model.Q(dt)
    R = measurement_model.R()

    F_t = torch.tensor(F, dtype=torch.float64)
    H_t = torch.tensor(H, dtype=torch.float64)
    Q_t = torch.tensor(Q, dtype=torch.float64)
    R_t = torch.tensor(R, dtype=torch.float64)

    diff_kf = DifferentiableKF(F_t, H_t, Q_t)

    # Batch size is 1 and R is fixed for all timesteps

    z_seq = torch.tensor(z, dtype=torch.float64).unsqueeze(0)
    R_seq = R_t.unsqueeze(0).repeat(K, 1, 1).unsqueeze(0)

    x0_t = torch.tensor(x0_filter, dtype=torch.float64).unsqueeze(0)
    P0_t = torch.tensor(P0, dtype=torch.float64).unsqueeze(0)

    x_seq, P_seq = diff_kf(z_seq, R_seq, dt, x0_t, P0_t)

    # Convert NumPy reference to torch for comparison
    x_ref_t = torch.tensor(x_ref, dtype=torch.float64).unsqueeze(0)
    P_ref_t = torch.tensor(P_ref, dtype=torch.float64).unsqueeze(0)


    assert torch.allclose(x_seq[0], x_ref_t[0], atol=1e-6), f"State mismatch"
    assert torch.allclose(P_seq[0], P_ref_t[0], atol=1e-6), f"Covariance mismatch"


def test_diff_kf_P_symmetric():
    """
    Test that the updated covariance matrix P is symmetric after each update step in diff_kf.
    """

    rng = np.random.default_rng(42)

    dt = 0.1
    K = 10
    x0_true = np.array([0, 0, 1, 1]) # px, py, vx, vy
    

    # Constant-velocity trajectory
    times = np.arange(1, K+1) * dt

    trajectory = np.column_stack([
        x0_true[0] + x0_true[2] * times,
        x0_true[1] + x0_true[3] * times,
    ])

    sigma_pos = 20
    z = trajectory + rng.normal(0.0, sigma_pos, size=(K, 2))

    # Filter setup
    motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
    measurement_model = LinearPosition2D(sigma_pos=sigma_pos) 

    x0_filter = np.array([.5, .3, 5, 10]) # Initial state uncertainty (random numbers I've chosen)
    P0 = np.array([[10, 0, 0, 0], 
                [0, 12, 0, 0], 
                [0, 0, 34, 0], 
                [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)

    # Extract F, H, Q, R matrices
    F = motion_model.F(x0_filter, dt)
    H = measurement_model.H(x0_filter)
    Q = motion_model.Q(dt)
    R = measurement_model.R()

    F_t = torch.tensor(F, dtype=torch.float64)
    H_t = torch.tensor(H, dtype=torch.float64)
    Q_t = torch.tensor(Q, dtype=torch.float64)
    R_t = torch.tensor(R, dtype=torch.float64)

    diff_kf = DifferentiableKF(F_t, H_t, Q_t)

    # Batch size is 1 and R is fixed for all timesteps

    z_seq = torch.tensor(z, dtype=torch.float64).unsqueeze(0)
    R_seq = R_t.unsqueeze(0).repeat(K, 1, 1).unsqueeze(0)

    x0_t = torch.tensor(x0_filter, dtype=torch.float64).unsqueeze(0)
    P0_t = torch.tensor(P0, dtype=torch.float64).unsqueeze(0)

    x_seq, P_seq = diff_kf(z_seq, R_seq, dt, x0_t, P0_t)

    # Check symmetry of P at each timestep
    for t in range(K):
        P_t = P_seq[0, t]
        assert torch.allclose(P_t, P_t.transpose(-2, -1), atol=1e-6), f"Covariance matrix P is not symmetric at timestep {t}"

def test_diff_kf_gradients_flow():
    """
    Test that gradients flow through the differentiable Kalman filter.
    """
    
    rng = np.random.default_rng(42)
    
    dt = 0.1
    K = 10
    x0_true = np.array([0, 0, 1, 1]) # px, py, vx, vy
    

    # Constant-velocity trajectory
    times = np.arange(1, K+1) * dt

    trajectory = np.column_stack([
        x0_true[0] + x0_true[2] * times,
        x0_true[1] + x0_true[3] * times,
    ])

    sigma_pos = 20
    z = trajectory + rng.normal(0.0, sigma_pos, size=(K, 2))

    # Filter setup
    motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
    measurement_model = LinearPosition2D(sigma_pos=sigma_pos) 

    x0_filter = np.array([.5, .3, 5, 10]) # Initial state uncertainty (random numbers I've chosen)
    P0 = np.array([[10, 0, 0, 0], 
                [0, 12, 0, 0], 
                [0, 0, 34, 0], 
                [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)

    # Reference KF
    kf = KalmanFilter(motion_model, measurement_model, x0=x0_filter, P0=P0)

    x_ref = []
    P_ref = []

    for t in range(K):
        kf.predict(dt)
        kf.update(z[t])

        x_ref.append(kf.x.copy())
        P_ref.append(kf.P.copy())

    x_ref = np.stack(x_ref)
    P_ref = np.stack(P_ref)

    # Extract F, H, Q, R matrices
    F = motion_model.F(x0_filter, dt)
    H = measurement_model.H(x0_filter)
    Q = motion_model.Q(dt)
    R = measurement_model.R()

    F_t = torch.tensor(F, dtype=torch.float64)
    H_t = torch.tensor(H, dtype=torch.float64)
    Q_t = torch.tensor(Q, dtype=torch.float64)
    R_t = torch.tensor(R, dtype=torch.float64, requires_grad=True)

    diff_kf = DifferentiableKF(F_t, H_t, Q_t)

    # Batch size is 1 and R is fixed for all timesteps

    z_seq = torch.tensor(z, dtype=torch.float64, requires_grad=True).unsqueeze(0)
    z_seq.retain_grad()  # Retain gradient for z_seq
    R_seq = R_t.unsqueeze(0).repeat(K, 1, 1).unsqueeze(0)
    R_seq.retain_grad()  # Retain gradient for R_seq

    x0_t = torch.tensor(x0_filter, dtype=torch.float64).unsqueeze(0)
    P0_t = torch.tensor(P0, dtype=torch.float64).unsqueeze(0)

    x_seq, P_seq = diff_kf(z_seq, R_seq, dt, x0_t, P0_t)

    loss = x_seq.sum() + P_seq.sum()
    loss.backward()

    # Check that z_seq grad exists and is finite
    assert z_seq.grad is not None, "z_seq did not receive a gradient"
    assert torch.isfinite(z_seq.grad).all(), "z_seq gradient is not a real number"

    # Check that R_t grad exists and is finite
    assert R_seq.grad is not None, "R_t did not receive a gradient"
    assert torch.isfinite(R_seq.grad).all(), "R_t gradient is not a real number"

def test_diff_kf_batch_independence():
    """
    Test that two different trajectories in the same batch are processed independently
    """

    # Different random seeds for two different trajectories (noise differs)
    rng_A = np.random.default_rng(42)
    rng_B = np.random.default_rng(43)

    dt = 0.1
    K = 10
    x0_true = np.array([0, 0, 1, 1]) # px, py, vx, vy
    
    # Constant-velocity trajectory
    times = np.arange(1, K+1) * dt

    trajectory = np.column_stack([
        x0_true[0] + x0_true[2] * times,
        x0_true[1] + x0_true[3] * times,
    ])

    sigma_pos = 20
    z_A = trajectory + rng_A.normal(0.0, sigma_pos, size=(K, 2))
    z_B = trajectory + rng_B.normal(0.0, sigma_pos, size=(K, 2))

    # Filter setup
    motion_model = ConstantVelocity2D(sigma_a=0.0) # No process noise
    measurement_model = LinearPosition2D(sigma_pos=sigma_pos) 

    x0_filter = np.array([.5, .3, 5, 10]) # Initial state uncertainty (random numbers I've chosen)
    P0 = np.array([[10, 0, 0, 0], 
                [0, 12, 0, 0], 
                [0, 0, 34, 0], 
                [0, 0, 0, 35]]) # Initial uncertainty (random numbers I've chosen)

    # Extract F, H, Q, R matrices
    F = motion_model.F(x0_filter, dt)
    H = measurement_model.H(x0_filter)
    Q = motion_model.Q(dt)
    R = measurement_model.R()

    F_t = torch.tensor(F, dtype=torch.float64)
    H_t = torch.tensor(H, dtype=torch.float64)
    Q_t = torch.tensor(Q, dtype=torch.float64)
    R_t = torch.tensor(R, dtype=torch.float64)

    diff_kf = DifferentiableKF(F_t, H_t, Q_t)

    # Batch size is 1 and R is fixed for all timesteps

    z_seq_A = torch.tensor(z_A, dtype=torch.float64).unsqueeze(0)
    z_seq_B = torch.tensor(z_B, dtype=torch.float64).unsqueeze(0)

    R_seq = R_t.unsqueeze(0).repeat(K, 1, 1).unsqueeze(0)

    x0_t = torch.tensor(x0_filter, dtype=torch.float64).unsqueeze(0)
    P0_t = torch.tensor(P0, dtype=torch.float64).unsqueeze(0)

    x_seq_A, P_seq_A = diff_kf(z_seq_A, R_seq, dt, x0_t, P0_t)
    x_seq_B, P_seq_B = diff_kf(z_seq_B, R_seq, dt, x0_t, P0_t)

    # Combine the two trajectories into a batch
    z_seq_batch = torch.cat((z_seq_A, z_seq_B), dim=0)
    x_seq_batch, P_seq_batch = diff_kf(z_seq_batch, R_seq.repeat(2, 1, 1, 1), dt, x0_t.repeat(2, 1), P0_t.repeat(2, 1, 1))

    # Check that the outputs for each trajectory in the batch match the independent runs
    assert torch.allclose(x_seq_batch[0], x_seq_A[0], atol=1e-6), "Trajectory A in batch does not match independent run"
    assert torch.allclose(x_seq_batch[1], x_seq_B[0], atol=1e-6), "Trajectory B in batch does not match independent run"


# To run: pytest tests/test_diff_kf.py -v