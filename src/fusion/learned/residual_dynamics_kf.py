import torch
import torch.nn as nn
import numpy as np

from fusion.learned.dynamics import ResidualDynamicsNet
from fusion.learned.diff_kf import DifferentiableKF
from tracking.models.measurement import LinearPosition2D
from tracking.models.motion import ConstantVelocity2D

class ResidualDynamicsKF(nn.Module):
    """
    Differentiable KF whose per-step dynamics correction and process noise 
    come from a trained ResidualDynamicsNet
    """

    def __init__(self, dynamics_net: ResidualDynamicsNet, F: torch.Tensor, H: torch.Tensor, R: torch.Tensor, dt: float) -> None:

        super().__init__()
        self.dynamics_net = dynamics_net
        self.dt = dt
        self.register_buffer('F', F) # Motion model state transition matrix
        self.register_buffer('H', H) # Measurement model selection matrix
        self.register_buffer('R', R) # Measurement Noise covariance matrix
        # Q is a placeholder as it now comes from dynamics_net
        self.diff_kf = DifferentiableKF(F=F, H=H, Q=torch.zeros_like(F))

    def forward(self, z_seq, x0, P0, h0=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the differentiable Kalman filter with ResidualDynamicsNet
        In: z_seq (B,K,dim_z), measurement sequence; 
            x0 (B, dim_x), initial state;
            P0 (B, dim_x, dim_x), initial error covariance;
            dt (float), time step;
            h0 (B,hidden_dim)| None, initial hidden state
        Out: x_seq (B, K, dim_x), sequence of estimated states;
             P_seq (B, K, dim_x, dim_x), sequence of estimated error covariances
             h_seq (B, K, hidden_dim), sequence of hidden states
        """

        # Create R_seq
        B, K = z_seq.shape[:2]
        R_seq = self.R.unsqueeze(0).unsqueeze(0).expand(B, K, -1, -1)

        x_seq, P_seq, h_seq = self.diff_kf(z_seq, R_seq, self.dt, x0, P0, dynamics_fn=self.dynamics_net, h0=h0)

        return (x_seq, P_seq, h_seq)

def build_residual_dynamics_kf(config: dict) -> nn.Module:
    """ Builder function for load_checkpoint for residual_dynamics_kf model
    In: config, configuration variables
    Out: model
    """

    motion_model = ConstantVelocity2D(sigma_a=0.0) #sigma_a is irrelevant as F doesnt use it
    measurement_model = LinearPosition2D(sigma_pos=config['maneuver']['sigma_pos'])

    dummy_state = np.zeros(4, dtype=np.float32)

    dt = config['scenario']['dt']

    F = torch.as_tensor(motion_model.F(dummy_state, dt), dtype=torch.float32)
    H = torch.as_tensor(measurement_model.H(dummy_state), dtype=torch.float32)
    R = torch.as_tensor(measurement_model.R(), dtype=torch.float32)

    P0_fixed = torch.as_tensor(config["initialization"]["P0"], dtype=torch.float32)
    x0 = torch.zeros(4, dtype=torch.float32)

    # Define residual dynamics model
    dynamics_net = ResidualDynamicsNet(dim_x=4, 
                                    hidden_dim=config['dynamics_net']['hidden_dim'],
                                    sigma_scale=config['dynamics_net']['sigma_scale'], 
                                    min_diag=config['dynamics_net']['min_diag'])
    
    model = ResidualDynamicsKF(dynamics_net, F, H, R, dt=dt)

    return model