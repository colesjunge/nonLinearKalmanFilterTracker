import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion.learned.dynamics import ResidualDynamicsNet


def test_output_shapes():
    """
    Assert delta is (B, dim_x), L is (B, dim_x, dim_x), h_next is (B, hidden_dim)
    """
    
    dim_x = 4
    hidden_dim = 64
    sigma_scale = 1.0
    min_diag = 1e-3
    B = 4

    model = ResidualDynamicsNet(dim_x, hidden_dim, sigma_scale, min_diag)

    x = torch.randn(B, dim_x)
    dt = .1
    h = torch.randn(B, hidden_dim)

    delta, L, h_next = model.forward(x, dt, h)

    assert delta.shape == (B, dim_x), "Output shape not (B, dim_x) for delta"
    assert L.shape == (B, dim_x, dim_x), "Output shape not (B, dim_x, dim_x) for L"
    assert h_next.shape == (B, hidden_dim), "Output shape not (B, hidden_dim) for h_next"

def test_zero_init_gives_zero_delta():
    """
    Assert delta is exactly all-zeros for an arbitrary batch of x/dt/h on an untrained network
    """

    dim_x = 4
    hidden_dim = 64
    sigma_scale = 1.0
    min_diag = 1e-3
    B = 4

    model = ResidualDynamicsNet(dim_x, hidden_dim, sigma_scale, min_diag)

    x = torch.randn(B, dim_x)
    dt = .1
    h = torch.randn(B, hidden_dim)

    delta, L, h_next = model.forward(x, dt, h)

    assert  torch.all(delta == 0), "Delta contains non-zero elements"


def test_L_diagonal_positive():
    """
    Assert L positive definite
    """

    dim_x = 4
    hidden_dim = 64
    sigma_scale = 1.0
    min_diag = 1e-3
    B = 4

    model = ResidualDynamicsNet(dim_x, hidden_dim, sigma_scale, min_diag)

    x = torch.randn(B, dim_x)
    dt = .1
    h = torch.randn(B, hidden_dim)

    delta, L, h_next = model.forward(x, dt, h)

    assert torch.all(torch.diagonal(L, dim1=-2, dim2=-1) > 0), "L not positive definite"

def test_gradients_flow():
    """
    Assert every parameter in gru_cell, delta_head, and covariance_head has a non-None, finite .grad
    """

    dim_x = 4
    hidden_dim = 64
    sigma_scale = 1.0
    min_diag = 1e-3
    B = 4

    model = ResidualDynamicsNet(dim_x, hidden_dim, sigma_scale, min_diag)

    x = torch.randn(B, dim_x)
    dt = .1
    h = torch.randn(B, hidden_dim)

    delta, L, h_next = model.forward(x, dt, h)

    loss = F.mse_loss(delta, torch.zeros_like(delta)) + F.mse_loss(L, torch.zeros_like(L))
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None, "Parameter did not receive a gradient"
        assert torch.isfinite(param.grad).all(), "Parameter gradient is not a real number"
    
    

def test_h_none_vs_explicit_zeros():
    """
    Assert calling forward with h = None is same as h = torch.zeros(B, hidden_dim)
    """

    dim_x = 4
    hidden_dim = 64
    sigma_scale = 1.0
    min_diag = 1e-3
    B = 4

    model = ResidualDynamicsNet(dim_x, hidden_dim, sigma_scale, min_diag)

    x = torch.randn(B, dim_x)
    dt = .1
    h = torch.zeros(B, hidden_dim)

    delta_none, L_none, h_next_none = model.forward(x, dt)
    delta_explicit, L_explicit, h_next_explicit = model.forward(x, dt, h)

    assert torch.equal(delta_none, delta_explicit), "Deltas do not match"
    assert torch.equal(L_none, L_explicit), "Ls do not match"
    assert torch.equal(h_next_none, h_next_explicit), "h_nexts do not match"
    

# To run: pytest tests/test_dynamics.py -v