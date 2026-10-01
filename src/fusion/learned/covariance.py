import torch
import torch.nn as nn
from fusion.learned.denoiser import Denoiser

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


