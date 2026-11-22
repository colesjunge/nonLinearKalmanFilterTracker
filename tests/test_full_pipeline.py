import numpy as np
import torch
from tracking.models.measurement import LinearPosition2D
from tracking.models.motion import ConstantVelocity2D
from fusion.learned.denoiser import Denoiser
from fusion.learned.dynamics import ResidualDynamicsNet
from fusion.learned.full_pipeline import FullPipelineModel



# Helper for tests
def make_full_pipeline_model(B=2, dim_x=4, hidden_dim=16, dim_z=2, dim_pos=2, hidden_channels=8, n_layers=2, kernel_size=3, W=10, dtype=torch.float64):

    dt = .1
    sigma_pos = 20 # arbitrary

    dummy_x = np.zeros(4)

    F = ConstantVelocity2D(sigma_a=0.0).F(dummy_x, dt)
    F_t = torch.tensor(F, dtype=dtype)
    H = LinearPosition2D(sigma_pos).H(dummy_x)
    H_t = torch.tensor(H, dtype=dtype)

    denoiser = Denoiser(dim_z=dim_z, 
                        dim_pos=dim_pos, 
                        hidden_channels=hidden_channels, 
                        n_layers=n_layers, 
                        kernel_size=kernel_size)

    dynamics_net = ResidualDynamicsNet(dim_x=dim_x, 
                                       hidden_dim=hidden_dim, 
                                       sigma_scale=1.0, 
                                       min_diag=1e-3)

    model = FullPipelineModel(denoiser, 
                              sigma_scale=1.0, 
                              min_diag=1e-3, 
                              dynamics_net=dynamics_net, 
                              F=F_t, 
                              H=H_t, 
                              dt=dt)
    model.to(dtype=dtype)

    x0 = torch.zeros(B, dim_x, dtype=dtype)
    diag = torch.tensor([1.5, 2.0, 3.5, 5], dtype=dtype) 
    P0 = batched_diag = torch.diag(diag).unsqueeze(0).expand(B, -1, -1)
    h0 = torch.zeros(B, hidden_dim, dtype=dtype)

    return model, x0, P0, h0, dt


def test_full_pipeline_output_shapes():
    """
    Assert output shapes are correct for FullPipelineModel
    """
    B = 2
    T = 10
    W = 10
    dim_z = 2
    dim_x = 4
    dtype = torch.float64
    hidden_dim = 16

    window_seq = torch.randn(B, T, W, dim_z, dtype=dtype)

    model, x0, P0, h0, dt = make_full_pipeline_model(B=B, dim_x=dim_x, dim_z=dim_z, hidden_dim=hidden_dim, W=W)

    x_seq, P_seq, h_seq = model(window_seq, x0, P0, h0)

    assert x_seq.shape == (B,T,dim_x), "x_seq incorrect dimensions"
    assert P_seq.shape == (B,T,dim_x,dim_x), "P_seq incorrect dimensions"
    assert h_seq.shape == (B,T,hidden_dim), "h_seq incorrect dimensions"


def test_full_pipeline_P_psd():
    """
    Assert P is positive semi-definite
    """
    
    B = 2
    T = 10
    W = 10
    dim_z = 2
    dim_x = 4
    dtype = torch.float64
    hidden_dim = 16

    window_seq = torch.randn(B, T, W, dim_z, dtype=dtype)

    model, x0, P0, h0, dt = make_full_pipeline_model(B=B, dim_x=dim_x, dim_z=dim_z, hidden_dim=hidden_dim, W=W)

    x_seq, P_seq, h_seq = model(window_seq, x0, P0, h0)


    for b in range(B):
        for t in range(T):
            eigvals = torch.linalg.eigvalsh(P_seq[b, t])
            assert torch.all(eigvals > -1e-8), "P is not semidefinite"


def test_full_pipeline_gradients_flow():
    """
    Assert denoiser has unfrozen gradients
    """
    
    B = 2
    T = 10
    W = 10
    dim_z = 2
    dim_x = 4
    dtype = torch.float64
    hidden_dim = 16

    window_seq = torch.randn(B, T, W, dim_z, dtype=dtype)

    model, x0, P0, h0, dt = make_full_pipeline_model(B=B, dim_x=dim_x, dim_z=dim_z, hidden_dim=hidden_dim, W=W)

    x_seq, P_seq, h_seq = model(window_seq, x0, P0, h0)

    loss = x_seq.sum() + P_seq.sum() + h_seq.sum()

    loss.backward()

    # Model parameters
    model_grads = [
        param.grad
        for _, param in model.named_parameters()
        if param.requires_grad
    ]

    assert all(grad is not None for grad in model_grads), "Some gradients are None"
    assert all(torch.isfinite(grad).all() for grad in model_grads), "Some gradients are inf"

    # Verify at least one parameter receives a nonzero gradient
    assert any(torch.any(grad != 0) for grad in model_grads), "No paramters receive nonzero gradients"
    

    # Denoiser parameters
    denoiser_grads = [
            param.grad
            for param in model.adaptive_denoiser.denoiser.parameters()
        ]

    assert all(grad is not None for grad in denoiser_grads), "Some gradients are None"
    assert all(torch.isfinite(grad).all() for grad in denoiser_grads), "Some gradients are inf"

    # Verify at least one parameter receives a nonzero gradient
    assert any(torch.any(grad != 0) for grad in denoiser_grads), "No paramters receive nonzero gradients"


def test_full_pipeline_causal_window_usage():
    """
    Assert that adaptive denoiser still follows causality
    """
    
    B = 2
    T = 10
    W = 10
    dim_z = 2
    dim_x = 4
    dtype = torch.float64
    hidden_dim = 16

    window_seq = torch.randn(B, T, W, dim_z, dtype=dtype)

    model, x0, P0, h0, dt = make_full_pipeline_model(B=B, dim_x=dim_x, dim_z=dim_z, hidden_dim=hidden_dim, W=W)

    x_seq_A, P_seq_A, h_seq_A = model(window_seq, x0, P0, h0)

    window_seq_per = window_seq.clone()

    window_seq_per[:, T-1] += torch.randn_like(window_seq_per[:, T-1])

    x_seq_B, P_seq_B, h_seq_B = model(window_seq_per, x0, P0, h0)

    assert torch.equal(x_seq_A[:, :T-1], x_seq_B[:, :T-1]), "Earlier state estimates changed"
    assert torch.equal(P_seq_A[:, :T-1], P_seq_B[:, :T-1]), "Earlier covariance estimates changed"
    assert torch.equal(h_seq_A[:, :T-1], h_seq_B[:, :T-1]), "Earlier hidden states changed"
    assert not torch.equal(x_seq_A[:, T-1], x_seq_B[:, T-1]), "Final state estimate did not change"


# To run: pytest tests/test_full_pipeline.py -v

