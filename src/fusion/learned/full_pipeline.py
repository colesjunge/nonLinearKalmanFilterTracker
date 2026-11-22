import torch
import torch.nn as nn
import numpy as np

from fusion.learned.denoiser import Denoiser
from fusion.learned.covariance import AdaptiveDenoiser, cov_from_chol
from fusion.learned.dynamics import ResidualDynamicsNet
from fusion.learned.diff_kf import DifferentiableKF
from tracking.models.motion import ConstantVelocity2D
from tracking.models.measurement import LinearPosition2D


class FullPipelineModel(nn.Module):
    """
    Differentiable KF with adaptive denoiser and residual dynamics correction
    """


    def __init__(self, denoiser: Denoiser, sigma_scale: float, min_diag: float, dynamics_net: ResidualDynamicsNet, F: torch.Tensor, H: torch.Tensor, dt: float) -> None:

        super().__init__()
        self.adaptive_denoiser = AdaptiveDenoiser(denoiser, sigma_scale, min_diag)
        
        # For this model adaptive denoiser needs trunk unfrozen so denoiser gets retrained
        for param in self.adaptive_denoiser.denoiser.parameters():
            param.requires_grad = True
        self.adaptive_denoiser.train()

        self.dynamics_net = dynamics_net
        self.diff_kf = DifferentiableKF(F=F, H=H, Q=torch.zeros_like(F))
        self.dt = dt

    def forward(self, window_seq: torch.Tensor, x0: torch.Tensor, P0: torch.Tensor, h0: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Per window, runs the denoiser trunk for features, derives the position estimate and R_theta via the covariance head
        Then feeds into one predict/update step of DifferentiableKF
        The predict step applies residual dynamics correction
        In: window_seq (B, T, W, 2), T consecutive raw-measurement windows;
           x0 (B,4), Initial state estimation; 
           P0 (B,4,4) Initial error covariance; 
           h0 (B, hidden_dim) | None, Initial hidden state (Doesn't exist for new tracks)
        Out: x_seq (B, T, dim_x), sequence of estimated states;
             P_seq (B, T, dim_x, dim_x), sequence of estimated error covariances
             h_seq (B, T, hidden_dim), sequence of hidden states
        """

        B, T, W, _ = window_seq.shape

        pos_seq = []
        L_seq = []

        # Run adaptive denoiser to get denoised positions and Cholesky factors
        for t in range(T):

            pos_t, L_t = self.adaptive_denoiser(window_seq[:, t])

            pos_seq.append(pos_t) # (B, 2)
            L_seq.append(L_t) # (B, 2, 2)

        pos_seq = torch.stack(pos_seq, dim=1) # (B, T, 2)
        L_seq = torch.stack(L_seq, dim=1) # (B, T, 2, 2)

        # Convert cholesky factors to covariance
        R_seq = cov_from_chol(L_seq) # (B, T, 2, 2)

        # Use this R_seq for diff_kf
        x_seq, P_seq, h_seq = self.diff_kf(pos_seq, R_seq, self.dt, x0, P0, dynamics_fn=self.dynamics_net, h0=h0)

        return x_seq, P_seq, h_seq


def build_full_pipeline_model(config: dict) -> nn.Module:
    """ Builder function for load_checkpoint for full_pipeline_model
    In: config, configuration variables
    Out: model
    """

    # Set up config variables
    dt = config["scenario"]["dt"]

    hidden_channels = config["model"]["hidden_channels"]
    n_layers = config["model"]["n_layers"]
    kernel_size = config["model"]["kernel_size"]
    dim_z = config["model"]["dim_z"]
    dim_pos = config["model"]["dim_pos"]

    hidden_dim = config["dynamics_net"]["hidden_dim"]
    sigma_scale_dn = config["dynamics_net"]["sigma_scale"]
    min_diag_dn = config["dynamics_net"]["min_diag"]

    sigma_scale_cov = config["covariance"]["sigma_scale"]
    min_diag_cov = config["covariance"]["min_diag"]

    dtype = torch.float32

    # Create Denoiser
    denoiser = Denoiser(dim_z=dim_z,
                        dim_pos=dim_pos,
                        hidden_channels=hidden_channels,
                        n_layers=n_layers,
                        kernel_size=kernel_size)

    # Compute F / H
    dummy_x = np.zeros(4, dtype=np.float32)
    F = torch.as_tensor(ConstantVelocity2D(sigma_a=0.0).F(dummy_x, dt), dtype=dtype)
    H = torch.as_tensor(LinearPosition2D(sigma_pos=1.0).H(dummy_x), dtype=dtype)

    # Define residual dynamics model
    dynamics_net = ResidualDynamicsNet(dim_x=4, 
                                        hidden_dim=hidden_dim,
                                        sigma_scale=sigma_scale_dn, 
                                        min_diag=min_diag_dn)
    
    model = FullPipelineModel(denoiser, sigma_scale_cov, min_diag_cov, dynamics_net, F, H, dt)

    return model
        

def build_full_pipeline_bearing_model(config: dict) -> nn.Module:
    """ Builder function for load_checkpoint for full_pipeline_bearing_model
    In: config, configuration variables
    Out: model
    """

    # Set up config variables
    dt = config["scenario_bearing"]["dt"]

    hidden_channels = config["model_bearing"]["hidden_channels"]
    n_layers = config["model_bearing"]["n_layers"]
    kernel_size = config["model_bearing"]["kernel_size"]
    dim_z = config["model_bearing"]["dim_z"]
    dim_pos = config["model_bearing"]["dim_pos"]

    hidden_dim = config["dynamics_net"]["hidden_dim"]
    sigma_scale_dn = config["dynamics_net"]["sigma_scale"]
    min_diag_dn = config["dynamics_net"]["min_diag"]

    sigma_scale_cov = config["covariance_bearing"]["sigma_scale"]
    min_diag_cov = config["covariance_bearing"]["min_diag"]

    dtype = torch.float32

    # Create Denoiser
    denoiser = Denoiser(dim_z=dim_z,
                        dim_pos=dim_pos,
                        hidden_channels=hidden_channels,
                        n_layers=n_layers,
                        kernel_size=kernel_size)

    # Compute F / H
    dummy_x = np.zeros(4, dtype=np.float32)
    F = torch.as_tensor(ConstantVelocity2D(sigma_a=0.0).F(dummy_x, dt), dtype=dtype)
    H = torch.as_tensor(LinearPosition2D(sigma_pos=1.0).H(dummy_x), dtype=dtype)

    # Define residual dynamics model
    dynamics_net = ResidualDynamicsNet(dim_x=4, 
                                        hidden_dim=hidden_dim,
                                        sigma_scale=sigma_scale_dn, 
                                        min_diag=min_diag_dn)
    
    model = FullPipelineModel(denoiser, sigma_scale_cov, min_diag_cov, dynamics_net, F, H, dt)

    return model

