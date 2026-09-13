import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from fusion.learned.denoiser import Denoiser


def test_denoiser_output_shape():
    """
    Test that output shape is (B, dim_pos)
    """

    B = 4
    W = 10
    dim_z = 2
    dim_pos = 2
    hidden_channels = 8
    n_layers = 4
    kernel_size = 3

    model = Denoiser(dim_z=dim_z, dim_pos=dim_pos, hidden_channels=hidden_channels, n_layers=n_layers, kernel_size=kernel_size)

    z_window = torch.randn(B, W, dim_z)

    output = model(z_window)

    assert output.shape == (B, dim_pos), "Output not correct shape"


def test_causality():
    """
    Test that causality is properly implemented
    Runs _conv_features on full length N sequence
    Then run on first N-1 positions
    Assert that they both agree exactly on their shared positions
    """
    B = 4
    N = 20
    dim_z = 2
    dim_pos = 2
    hidden_channels = 8
    n_layers = 4
    kernel_size = 3
    
    model = Denoiser(dim_z=dim_z, dim_pos=dim_pos, hidden_channels=hidden_channels, n_layers=n_layers, kernel_size=kernel_size)

    # Full 20 timestep sequence
    full_sequence = torch.randn(B, N, dim_z)
    full_output = model._conv_features(full_sequence)

    # First 19 steps sequence
    truncated_sequence = full_sequence[:, :N-1, :]
    truncated_output = model._conv_features(truncated_sequence)

    # First 19 of full output to compare (features and time get swapped)
    comparable_full_output = full_output[:, :, :N-1]

    assert torch.equal(comparable_full_output, truncated_output), "Causality broken as first N-1 outputs do not match"

def test_gradients_flow():
    """
    Create a small Denoiser and run one forward and backward pass
    Assert that every parameter received a finite (not inf or NaN) gradient 
    """

    B = 4
    W = 10
    dim_z = 2
    dim_pos = 2
    hidden_channels = 8
    n_layers = 4
    kernel_size = 3
    
    model = Denoiser(dim_z=dim_z, dim_pos=dim_pos, hidden_channels=hidden_channels, n_layers=n_layers, kernel_size=kernel_size)

    z_window = torch.randn(B, W, dim_z)
    target = torch.randn(B, dim_pos)

    # Forward pass, MSE loss, then backward
    predictions = model(z_window)
    loss = F.mse_loss(predictions, target)
    loss.backward()

    # Check that param grad exists and is finite
    for param in model.parameters():
        assert param.grad is not None, "Parameter did not receive a gradient"
        assert torch.isfinite(param.grad).all(), "Parameter gradient is not a real number"


# To run: pytest tests/test_learned_shapes.py -v

