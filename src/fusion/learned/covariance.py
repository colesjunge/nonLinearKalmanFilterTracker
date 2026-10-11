import torch
import torch.nn as nn
import numpy as np
from tracking.models.measurement import LinearPosition2D
from tracking.models.motion import ConstantVelocity2D
from fusion.learned.denoiser import Denoiser
from fusion.learned.diff_kf import DifferentiableKF

def gaussian_nll(pred: torch.Tensor, target: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    """ Calculates the mean negative log-likelihood over the batch
        loss = 0.5 · y_T*S_inv*y + 0.5*log det(S)
        S = L*L_T
        z = L_inv*y

    In: pred (B, dim_pos);
        target (B, dim_pos);
        L (B, dim_pos, dim_pos) lower triangular Cholesky factor for predicted 
    Out: scalar (mean NLL over batch)
    """

    y = pred - target  # (B, dim_pos)

    # Use solve_triangular to solve for z
    z = torch.linalg.solve_triangular(L, y.unsqueeze(-1), upper=False).squeeze(-1)  # (B, dim_pos)

    diag_L = torch.diagonal(L, dim1=-2, dim2=-1)  # (B, dim_pos)

    sum_square = (z ** 2).sum(dim=-1)          # (B,)
    sum_log_diag = torch.log(diag_L).sum(dim=-1)  # (B,)

    loss_per_sample = 0.5 * sum_square + sum_log_diag  # (B,)

    return loss_per_sample.mean()

def cov_from_chol(L: torch.Tensor) -> torch.Tensor:
    """
    Compute covariance from Cholesky factor
    In:  L (B, dim_pos, dim_pos) lower-triangular Cholesky factor
    Out: covariance (B, dim_pos, dim_pos)
    """

    return L @ L.mT

class CovarianceHead(nn.Module):
    """
    Maps trunk feature vector to a lower-triangular Cholesky factor L
    """

    def __init__(self, hidden_channels: int, dim_pos: int, sigma_scale: float, min_diag: float):

        super().__init__()
        self.hidden_channels = hidden_channels
        self.dim_pos = dim_pos
        self.sigma_scale = sigma_scale
        self.min_diag = min_diag

        n_lower_tri = dim_pos*(dim_pos+1)//2

        self.norm = nn.LayerNorm(hidden_channels) # Account for variable trajectory magnitudes
        self.linear = nn.Linear(hidden_channels, n_lower_tri)
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """ Map features to output lower-triangular Cholesky factor L
        In:  features (B, hidden_channels)
        Out: L (B, dim_pos, dim_pos) lower-triangular, positive diagonal
        """

        # Map the features to lower triangular values
        norm_features = self.norm(features)
        raw = self.linear(norm_features)

        # Create the empty lower-triangular matrix
        B = features.shape[0]
        L = torch.zeros(B, self.dim_pos, self.dim_pos, dtype=features.dtype, device=features.device)

        # Get lower-triangular indices
        rows, cols = torch.tril_indices(self.dim_pos, self.dim_pos)

        # Scatter raw values into L
        L[:, rows, cols] = raw

        # Transform only  diagonal entries
        diag_mask = rows == cols
        L[:, rows[diag_mask], cols[diag_mask]] = (torch.nn.functional.softplus(raw[:, diag_mask]) * self.sigma_scale + self.min_diag)

        return L

class AdaptiveDenoiser(nn.Module):
    """
    Wraps the trained, frozen Denoiser trunk with a trainable CovarianceHead
    """

    def __init__(self, denoiser: Denoiser, sigma_scale: float, min_diag: float):

        super().__init__()
        self.denoiser = denoiser
        self.sigma_scale = sigma_scale
        self.min_diag = min_diag

        # Freeze all denoiser parameters
        for param in denoiser.parameters():
            param.requires_grad = False
        denoiser.eval()

        self.covariance_head = CovarianceHead(hidden_channels=denoiser.hidden_channels, 
                                        dim_pos=denoiser.dim_pos, 
                                        sigma_scale=sigma_scale, 
                                        min_diag=min_diag)


    def forward(self, z_window: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Map casual window of raw measurements to denoised position estimate and trained covariance
        In:  z_window (B, W, dim_z)
        Out: pos (B, dim_pos), L (B, dim_pos, dim_pos)
        """

        x = self.denoiser._conv_features(z_window)

        features = x[:, :, -1]

        pos = self.denoiser.output_layer(features)

        L = self.covariance_head(features)

        return pos, L

def build_adaptive_denoiser(config: dict) -> nn.Module:
    """ Builder function for load_checkpoint for adaptive denoiser model
    In: config, configuration variables
    Out: adaptive denoiser
    """

    denoiser = Denoiser(dim_z=config['model']['dim_z'],
                            dim_pos=config['model']['dim_pos'], 
                            hidden_channels=config['model']['hidden_channels'], 
                            n_layers=config['model']['n_layers'], 
                            kernel_size=config['model']['kernel_size'])

    adaptive_denoiser = AdaptiveDenoiser(denoiser=denoiser, 
                                        sigma_scale=config['covariance']['sigma_scale'], 
                                        min_diag=config['covariance']['min_diag'])

    return adaptive_denoiser

class RolloutCovarianceModel(nn.Module):

    def __init__(self, 
                 denoiser: Denoiser, 
                 sigma_scale: float, 
                 min_diag: float, 
                 F: torch.Tensor, 
                 H: torch.Tensor, 
                 Q: torch.Tensor, 
                 dt: int):
        
        super().__init__()

        self.adaptive_denoiser = AdaptiveDenoiser(denoiser, sigma_scale, min_diag)
        self.diff_kf = DifferentiableKF(F, H, Q)

        self.dt = dt

    def forward(self, z_window_seq: torch.Tensor, x0: torch.Tensor, P0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ Forward pass through the adaptive denoiser and differentiable Kalman filter
        In:  z_window_seq (B, K, W, dim_z), sequence of raw measurement windows
             x0 (B, dim_x), initial state estimate
             P0 (B, dim_x, dim_x), initial error covariance

        Out: x_filtered (B, K, dim_x), sequence of estimated states;
             P_filtered (B, K, dim_x, dim_x), sequence of estimated error covariances
        """
        B, K, _, _ = z_window_seq.shape

        pos_seq = []
        L_seq = []

        for t in range(K):
            pos_t, L_t = self.adaptive_denoiser(z_window_seq[:, t])
            pos_seq.append(pos_t)
            L_seq.append(L_t)

        pos_seq = torch.stack(pos_seq, dim=1) # (B, K, dim_pos)
        L_seq = torch.stack(L_seq, dim=1) # (B, K, dim_pos, dim_pos)

        R_seq = cov_from_chol(L_seq) # (B, K, dim_pos, dim_pos)

        x0_expanded = x0.expand(B, -1) # (B, dim_x)
        P0_expanded = P0.expand(B, -1, -1) # (B, dim_x, dim_x)

        x_filtered, P_filtered = self.diff_kf(pos_seq, R_seq, self.dt, x0_expanded, P0_expanded)

        return x_filtered, P_filtered

def build_rollout_covariance_model(config: dict) -> nn.Module:
    """ Builder function for load_checkpoint for rollout covariance model
    In: config, configuration variables
    Out: rollout covariance model
    """

    denoiser = Denoiser(dim_z=config['model']['dim_z'],
                            dim_pos=config['model']['dim_pos'], 
                            hidden_channels=config['model']['hidden_channels'], 
                            n_layers=config['model']['n_layers'], 
                            kernel_size=config['model']['kernel_size'])

    motion_model = ConstantVelocity2D(sigma_a=0.0)
    measurement_model = LinearPosition2D(sigma_pos=1.0) # Sigma_pos is arbitrary here

    dummy_state = np.zeros(4, dtype=np.float32)

    dt = config['scenario']['dt']

    F = torch.as_tensor(motion_model.F(dummy_state, dt), dtype=torch.float32)
    H = torch.as_tensor(measurement_model.H(dummy_state), dtype=torch.float32)
    Q = torch.as_tensor(motion_model.Q(dt), dtype=torch.float32)

    rollout_cov_model = RolloutCovarianceModel(denoiser=denoiser, 
                                        sigma_scale=config['covariance']['sigma_scale'], 
                                        min_diag=config['covariance']['min_diag'],
                                        F=F,
                                        H=H, 
                                        Q=Q, 
                                        dt=dt)

    return rollout_cov_model

