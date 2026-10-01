import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion.learned.denoiser import Denoiser
from fusion.learned.covariance import gaussian_nll, cov_from_chol, CovarianceHead, AdaptiveDenoiser



def test_gaussian_nll__zero_case():
    """
    Test NLL is 0 when L = I and pred = target
    """

    L = torch.eye(2).unsqueeze(0)
    # pred = target; y = [0, 0]
    pred = torch.tensor([[5.0, 7.0]]) 
    target = torch.tensor([[5.0, 7.0]]) 

    nll = gaussian_nll(pred, target, L)

    assert torch.allclose(nll, torch.zeros(1)), "NLL not equal to 0"


def test_gaussian_nll_quadratic_case():
    """
    Test NLL computes correct value when L = I (pred ≠ target)
    """

    L = torch.eye(2).unsqueeze(0)
    pred = torch.tensor([[5.0, 7.0]])
    target = torch.tensor([[2.0, 3.0]])
    # y = [3.0, 4.0]
    # 0.5 * (3**2 + 4**2) = 12.5

    nll = gaussian_nll(pred, target, L)

    assert torch.allclose(nll, torch.tensor([12.5])), "NLL not equal to 12.5"

def test_gaussian_nll_log_case():
    """
    Test NLL computes correct value when pred = target (L ≠ I)
    """

    L = 2*torch.eye(2).unsqueeze(0)
    # pred = target; y = [0, 0]
    pred = torch.tensor([[5.0, 7.0]]) 
    target = torch.tensor([[5.0, 7.0]]) 

    nll = gaussian_nll(pred, target, L)

    assert torch.allclose(nll, 2*torch.log(torch.tensor(2.0))), "NLL not equal to 2log(2)"

def test_gaussian_nll_brute_force():
    """
    Test NLL with non-diagonal L (check is through brute force)
    """

    L = torch.tensor([[2,0],[0.5,1]]).unsqueeze(0)
    pred = torch.tensor([[5.0, 7.0]])
    target = torch.tensor([[2.0, 3.0]])

    nll = gaussian_nll(pred, target, L)
    
    # Brute Force
    y = pred - target 
    S = L @ L.mT # .mT to preserve batches
    brute_force_nll = 0.5 * torch.linalg.vecdot(y, torch.linalg.solve(S, y), dim=-1) + 0.5 * torch.logdet(S)

    assert torch.allclose(nll, brute_force_nll), "NLL not equal to brute force NLL"

def test_gaussian_nll_batch_correct():

    # NLL = 0
    batch_1_L = torch.eye(2).unsqueeze(0)
    batch_1_pred = torch.tensor([[5.0, 7.0]]) 
    batch_1_target = torch.tensor([[5.0, 7.0]]) 

    # NLL = 12.5
    batch_2_L = torch.eye(2).unsqueeze(0)
    batch_2_pred = torch.tensor([[5.0, 7.0]])
    batch_2_target = torch.tensor([[2.0, 3.0]])

    # NLL = 2log(2)
    batch_3_L = 2*torch.eye(2).unsqueeze(0)
    batch_3_pred = torch.tensor([[5.0, 7.0]]) 
    batch_3_target = torch.tensor([[5.0, 7.0]])

    # Calculated from prior tests
    true_nll = torch.tensor((0 + 12.5 + 2*torch.log(torch.tensor(2.0)).item()) / 3)

    L = torch.cat((batch_1_L, batch_2_L, batch_3_L), dim=0)
    pred = torch.cat((batch_1_pred, batch_2_pred, batch_3_pred), dim=0)
    target = torch.cat((batch_1_target, batch_2_target, batch_3_target), dim=0)

    nll = gaussian_nll(pred, target, L)

    assert torch.allclose(nll, true_nll), "NLL not equal to true NLL"


def test_gaussian_nll_overconfidence():
    """
    Test that small L results is larger loss than larger L for large y 
    """

    L1 = .5*torch.eye(2).unsqueeze(0)
    L2 = 5*torch.eye(2).unsqueeze(0)

    pred = torch.tensor([[55.0, 10.0]])
    target = torch.tensor([[12.0, 3.0]])

    nll_L1 = gaussian_nll(pred, target, L1)
    nll_L2 = gaussian_nll(pred, target, L2)

    assert nll_L2 < nll_L1, "Overconfidence not penalized"

def test_gaussian_nll_underconfidence():
    """
    Test that small L results is larger loss than larger L for y = 0
    """

    L1 = .5*torch.eye(2).unsqueeze(0)
    L2 = 5*torch.eye(2).unsqueeze(0)

    pred = torch.tensor([[5.0, 7.0]]) 
    target = torch.tensor([[5.0, 7.0]])

    nll_L1 = gaussian_nll(pred, target, L1)
    nll_L2 = gaussian_nll(pred, target, L2)

    assert nll_L2 > nll_L1, "Overconfidence not penalized"

def test_cov_from_chol_I():
    """
    Test that L = I results in I
    """

    L = torch.eye(2).unsqueeze(0)
    R = cov_from_chol(L)

    assert torch.allclose(L, R), "R is not I"

def test_cov_from_chol_diagonal():
    """
    Test that L diagonal results in squares of diagonal
    """

    L = torch.tensor([[2.0, 0.0], [0.0, 3.0]]).unsqueeze(0)
    R = cov_from_chol(L)
    R_true = torch.tensor([[4.0, 0.0], [0.0, 9.0]]) 

    assert torch.allclose(R_true, R), "R is not square of L"

def test_cov_from_chol_brute_force():
    """
    Test that L non diagonal matches brute force
    """

    L = torch.tensor([[2.0, 0.0], [5.0, 3.0]]) 
    R = cov_from_chol(L)
    R_brute_force = L@L.mT

    eigen_values = torch.linalg.eigvalsh(R)

    assert torch.allclose(R_brute_force, R), "R does not match brute force"
    assert (eigen_values > 0).all(), "R not positive-definite"

def test_cov_from_chol_batch_correct():
    """
    Test that L batches match independent batches
    """
     
    batch_1_L = torch.eye(2).unsqueeze(0)
    batch_2_L = torch.eye(2).unsqueeze(0)
    batch_3_L = 2*torch.eye(2).unsqueeze(0)

    R_L1 = cov_from_chol(batch_1_L)
    R_L2 = cov_from_chol(batch_2_L)
    R_L3 = cov_from_chol(batch_3_L)

    R_independent = torch.cat((R_L1, R_L2, R_L3), dim=0)

    L = torch.cat((batch_1_L, batch_2_L, batch_3_L), dim=0)

    R = cov_from_chol(L)

    assert torch.allclose(R, R_independent), "R not close to independently calculated R"

def test_cov_head_output_shape():
    """
    Test that output shape is (B, dim_pos, dim_pos)
    """

    B = 4
    dim_pos = 2
    hidden_channels = 8

    sigma_scale = 20
    min_diag = 1.0e-3

    model = CovarianceHead(hidden_channels = hidden_channels, dim_pos = dim_pos, sigma_scale = sigma_scale, min_diag = min_diag)

    feature = torch.randn(B, hidden_channels)

    L = model(feature)

    assert L.shape == (B, dim_pos, dim_pos), "Output not correct shape"

def test_cov_head_zero_upper_triangular():
    """
    Test that entry [0, 1] is 0 to ensure lower triangularity
    """

    B = 4
    dim_pos = 2
    hidden_channels = 8

    sigma_scale = 20
    min_diag = 1.0e-3

    model = CovarianceHead(hidden_channels = hidden_channels, dim_pos = dim_pos, sigma_scale = sigma_scale, min_diag = min_diag)

    feature = torch.randn(B, hidden_channels)

    L = model(feature)

    assert torch.allclose(L[:, 0, 1], torch.zeros(B)), "L not lower triangular"

def test_cov_head_positive_diagonal():
    """
    Test that diagonals are all positive to enforce positive definite L
    """

    B = 4
    dim_pos = 2
    hidden_channels = 8

    sigma_scale = 20
    min_diag = 1.0e-3

    model = CovarianceHead(hidden_channels = hidden_channels, dim_pos = dim_pos, sigma_scale = sigma_scale, min_diag = min_diag)

    feature = torch.randn(B, hidden_channels)

    L = model(feature)

    assert (torch.diagonal(L, dim1=-2, dim2=-1) > torch.zeros(B, 2)).all(), "L not positive definite"

def test_cov_head_gradients_flow():
    """
    Create a small Covariance Head and run one forward and backward pass
    Assert that every parameter received a finite (not inf or NaN) gradient 
    """
    
    B = 4
    dim_pos = 2
    hidden_channels = 8

    sigma_scale = 20
    min_diag = 1.0e-3

    model = CovarianceHead(hidden_channels = hidden_channels, dim_pos = dim_pos, sigma_scale = sigma_scale, min_diag = min_diag)

    feature = torch.randn(B, hidden_channels)
    target = torch.randn(B, dim_pos, dim_pos)

    L = model(feature)

    loss = F.mse_loss(L, target)
    loss.backward()

    # Check that param grad exists and is finite
    for param in model.parameters():
        assert param.grad is not None, "Parameter did not receive a gradient"
        assert torch.isfinite(param.grad).all(), "Parameter gradient is not a real number"

def test_adaptive_denoiser_output_shape():
    """
    Test that output shape is (B, dim_pos) for pos and (B, dim_pos, dim_pos) for L
    """

    B = 4
    W = 10
    dim_z = 2
    dim_pos = 2
    hidden_channels = 8
    n_layers = 4
    kernel_size = 3
    sigma_scale = 20
    min_diag = 1.0e-3

    denoiser = Denoiser(dim_z=dim_z, dim_pos=dim_pos, hidden_channels=hidden_channels, n_layers=n_layers, kernel_size=kernel_size)

    model = AdaptiveDenoiser(denoiser=denoiser, sigma_scale= sigma_scale, min_diag=min_diag)

    z_window = torch.randn(B, W, dim_z)

    pos, L = model(z_window)

    assert pos.shape == (B, dim_pos), "pos not correct shape"
    assert L.shape == (B, dim_pos, dim_pos), "L not correct shape"


def test_adaptive_denoiser_freeze():
    """
    Create a small Denoiser and run one forward and backward pass
    Assert that every parameter in covariance_head received a finite (not inf or NaN) gradient 
    Assert that every trunk parameter is None
    """

    B = 4
    W = 10
    dim_z = 2
    dim_pos = 2
    hidden_channels = 8
    n_layers = 4
    kernel_size = 3
    sigma_scale = 20
    min_diag = 1.0e-3
    
    denoiser = Denoiser(dim_z=dim_z, dim_pos=dim_pos, hidden_channels=hidden_channels, n_layers=n_layers, kernel_size=kernel_size)
    model = AdaptiveDenoiser(denoiser=denoiser, sigma_scale= sigma_scale, min_diag=min_diag)


    z_window = torch.randn(B, W, dim_z)
    target = torch.randn(B, dim_pos)

    # Forward pass, MSE loss, then backward
    predictions, L = model(z_window)
    loss = gaussian_nll(predictions, target, L)
    loss.backward()

    # Check that param grad are none for denoiser
    for param in model.denoiser.parameters():
        assert param.grad is None, "Denoiser parameter did receive a gradient"

    # Check that covariance_head param grad exists and is finite
    for param in model.covariance_head.parameters():
        assert param.grad is not None, "Covariance_head parameter did not receive a gradient"
        assert torch.isfinite(param.grad).all(), "Covariance_head parameter gradient is not a real number"
        
        
# To run: pytest tests/test_learned_covariance.py -v