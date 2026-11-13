import numpy as np
import torch
from tracking.filters.kalman import KalmanFilter
from tracking.models.measurement import LinearPosition2D
from tracking.models.motion import ConstantVelocity2D
from fusion.learned.diff_kf import DifferentiableKF
from fusion.learned.dynamics import ResidualDynamicsNet
from fusion.learned.residual_dynamics_kf import ResidualDynamicsKF

# Helper function to create residual kf, x0_t, P0_t, h0, dt for tests below
def make_residual_kf(B=1, dim_x=4, hidden_dim=64, dtype=torch.float64):

    # Filter parameters
    dt = 0.1
    sigma_pos = 20.0

    motion_model = ConstantVelocity2D(sigma_a=0.0)
    measurement_model = LinearPosition2D(sigma_pos=sigma_pos)

    x0_filter = np.array([0.5, 0.3, 5, 10])

    P0 = np.diag([10, 12, 34, 35])

    # Extract matrices
    F = motion_model.F(x0_filter, dt)
    H = measurement_model.H(x0_filter)
    R = measurement_model.R()

    F_t = torch.tensor(F, dtype=dtype)
    H_t = torch.tensor(H, dtype=dtype)
    R_t = torch.tensor(R, dtype=dtype)

    # Initialize learned dynamics
    dynamics_net = ResidualDynamicsNet(dim_x=dim_x, hidden_dim=hidden_dim, sigma_scale=1.0, min_diag=1e-3).to(dtype=dtype)

    # Initialize residual dynamics KF
    diff_kf = ResidualDynamicsKF(dynamics_net=dynamics_net, F=F_t, H=H_t, R=R_t, dt=dt)

    # Initial conditions
    x0_t = torch.tensor(x0_filter, dtype=dtype).unsqueeze(0).repeat(B, 1)

    P0_t = torch.tensor(P0, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)

    h0 = torch.zeros(B, hidden_dim, dtype=dtype)

    return diff_kf, x0_t, P0_t, h0, dt

def test_output_shapes():
    """
    Assert x_seq is (B, K, dim_x), P_seq is (B, K, dim_x, dim_x), h_seq is (B, K, hidden_dim)
    """

    rng = np.random.default_rng(42)

    # Matches default used in helper function
    dim_x = 4
    hidden_dim = 64 

    K = 10
    B = 1
    dt = 0.1

    x0_true = np.array([0, 0, 1, 1])

    # Generate constant-velocity trajectory
    times = np.arange(1, K+1) * dt

    trajectory = np.column_stack([
        x0_true[0] + x0_true[2] * times,
        x0_true[1] + x0_true[3] * times,
    ])

    sigma_pos = 20
    z = trajectory + rng.normal(0.0, sigma_pos, size=(K, 2))

    # Initialize residual KF
    diff_kf, x0_t, P0_t, h0, dt = make_residual_kf(B=B)

    z_seq = torch.tensor(z, dtype=torch.float64).unsqueeze(0)

    # Run residual dynamics KF
    x_seq, P_seq, h_seq = diff_kf(z_seq, x0_t, P0_t, h0)

    assert x_seq.shape == (B, K, dim_x), "Output shape not (B, K, dim_x) for delta"
    assert P_seq.shape == (B, K, dim_x, dim_x), "Output shape not (B, K, dim_x, dim_x) for L"
    assert h_seq.shape == (B, K, hidden_dim), "Output shape not (B, K, hidden_dim) for h_next"


def test_P_symmetric():
    """
    Test that the updated covariance matrix P is symmetric after each update step in residual_diff_kf.
    """

    rng = np.random.default_rng(42)

    K = 10
    B = 1
    dt = 0.1

    x0_true = np.array([0, 0, 1, 1])

    # Generate constant-velocity trajectory
    times = np.arange(1, K+1) * dt

    trajectory = np.column_stack([
        x0_true[0] + x0_true[2] * times,
        x0_true[1] + x0_true[3] * times,
    ])

    sigma_pos = 20
    z = trajectory + rng.normal(0.0, sigma_pos, size=(K, 2))

    # Initialize residual KF
    diff_kf, x0_t, P0_t, h0, dt = make_residual_kf(B=B)

    z_seq = torch.tensor(z, dtype=torch.float64).unsqueeze(0)

    # Run residual dynamics KF
    x_seq, P_seq, h_seq = diff_kf(z_seq, x0_t, P0_t, h0)

    # Check covariance symmetry at every timestep
    for t in range(K):
        P_t = P_seq[0, t]
        assert torch.allclose(P_t, P_t.transpose(-2, -1), atol=1e-6), f"Covariance matrix P is not symmetric at timestep {t}"

def test_diff_gradients_flow():
    """
    Test that gradients flow through the differentiable Kalman filter.
    """
    rng = np.random.default_rng(42)

    K = 10
    B = 1
    dt = 0.1

    x0_true = np.array([0, 0, 1, 1])

    # Generate constant-velocity trajectory
    times = np.arange(1, K+1) * dt

    trajectory = np.column_stack([
        x0_true[0] + x0_true[2] * times,
        x0_true[1] + x0_true[3] * times,
    ])

    sigma_pos = 20
    z = trajectory + rng.normal(0.0, sigma_pos, size=(K, 2))

    # Initialize residual KF
    diff_kf, x0_t, P0_t, h0, dt = make_residual_kf(B=B)

    z_seq = torch.tensor(z, dtype=torch.float64).unsqueeze(0).requires_grad_()

    # Run filter
    x_seq, P_seq, h_seq = diff_kf(z_seq, x0_t, P0_t, h0)

    # Construct loss
    loss = x_seq.sum() + P_seq.sum() + h_seq.sum()

    loss.backward()

    # Check measurement gradients
    assert z_seq.grad is not None
    assert torch.isfinite(z_seq.grad).all()

    # Check neural network gradients
    grads = [
        param.grad
        for param in diff_kf.dynamics_net.parameters()
        if param.requires_grad
    ]

    assert all(grad is not None for grad in grads), "Some gradients are None"
    assert all(torch.isfinite(grad).all() for grad in grads), "Some gradients are inf"

    # Verify at least one parameter receives a nonzero gradient
    assert any(torch.any(grad != 0) for grad in grads), "No paramters receive nonzero gradients"

def test_diff_batch_independence():
    """
    Test that two different trajectories in the same batch are processed independently
    """

    rng_A = np.random.default_rng(42)
    rng_B = np.random.default_rng(43)

    K = 10
    dt = 0.1

    x0_true = np.array([0, 0, 1, 1])

    # Generate constant-velocity trajectory
    times = np.arange(1, K+1) * dt

    trajectory = np.column_stack([
        x0_true[0] + x0_true[2] * times,
        x0_true[1] + x0_true[3] * times,
    ])

    sigma_pos = 20

    z_A = trajectory + rng_A.normal(0.0, sigma_pos, size=(K, 2))
    z_B = trajectory + rng_B.normal(0.0, sigma_pos, size=(K, 2))

    # Initialize residual KF
    diff_kf, x0_t, P0_t, h0, dt = make_residual_kf(B=1)

    # Individual trajectories
    z_seq_A = torch.tensor(z_A, dtype=torch.float64).unsqueeze(0)
    z_seq_B = torch.tensor(z_B, dtype=torch.float64).unsqueeze(0)

    x_seq_A, P_seq_A, h_seq_A = diff_kf(z_seq_A, x0_t, P0_t, h0)

    x_seq_B, P_seq_B, h_seq_B = diff_kf(z_seq_B, x0_t, P0_t, h0)

    # Combine trajectories into a batch
    z_seq_batch = torch.cat((z_seq_A, z_seq_B), dim=0)

    x0_batch = x0_t.repeat(2, 1)
    P0_batch = P0_t.repeat(2, 1, 1)
    h0_batch = h0.repeat(2, 1)

    x_seq_batch, P_seq_batch, h_seq_batch = diff_kf(z_seq_batch, x0_batch, P0_batch, h0_batch)

    # Compare individual runs against batched run
    assert torch.allclose(x_seq_batch[0], x_seq_A[0], atol=1e-6), "Trajectory A does not match independent run"
    assert torch.allclose(x_seq_batch[1], x_seq_B[0], atol=1e-6), "Trajectory B does not match independent run"

    assert torch.allclose(P_seq_batch[0], P_seq_A[0], atol=1e-6), "Covariance A does not match independent run"
    assert torch.allclose(P_seq_batch[1], P_seq_B[0], atol=1e-6), "Covariance B does not match independent run"

    assert torch.allclose(h_seq_batch[0], h_seq_A[0], atol=1e-6), "Hidden States A does not match independent run"
    assert torch.allclose(h_seq_batch[1], h_seq_B[0], atol=1e-6), "Hidden States B does not match independent run"

# To run: pytest tests/test_residual_dynamics_kf.py -v