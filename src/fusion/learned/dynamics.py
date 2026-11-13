import torch
import torch.nn as nn

from fusion.learned.covariance import CovarianceHead

class ResidualDynamicsNet(nn.Module):
    """
    Learned dyanmics network trained through DifferentiableKF
    Predicts a correction added to the CV prediction, the Cholesky factor of its own residual process noise
    Uses a GRU cell so correction can depend on recent track history 
    """


    def __init__(self, dim_x: int = 4, hidden_dim: int = 64, sigma_scale: float = 1.0, min_diag: float = 1e-3) -> None:
        """
        In: dim_x, state dimension (4 for [px, py, vx, vy]);
            hidden_dim, GRU hidden state size;
            sigma_scale, min_diag, passed through to the reused CovarianceHead
        """
        super().__init__()
        self.dim_x = dim_x
        self.hidden_dim = hidden_dim

        # Input size is dim_x +1 to account for x (dim_x) and dt (1)
        input_size = dim_x + 1
        # GRU for recurrent core
        self.gru_cell = nn.GRUCell(input_size=input_size, hidden_size=hidden_dim)
        # Map GRU's updated hidden state to additive correction delta
        self.delta_head = nn.Linear(hidden_dim, dim_x)
        # Set delta_head weights and biases to zero at construction so model is identical to plain CV prediction at start
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        # Use CovarianceHead to construct L (it is not position specific so can be reused)
        self.covariance_head = CovarianceHead(hidden_channels=hidden_dim, dim_pos=dim_x, sigma_scale=sigma_scale, min_diag=min_diag)


    def forward(self, x: torch.Tensor, dt: float, h: torch.Tensor | None = None)-> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict one step's correction to the CV prediction, given the current filtered state
        and the recurrent hidden state from the previous step.

        In:  x, (B, dim_x) current filtered state estimate;
            dt, seconds since last update;
            h, (B, hidden_dim) | None — recurrent hidden state; None for a
                newly-born track, where it's initialized to zeros
        Out: delta, (B, dim_x) additive correction to F @ x;
            L, (B, dim_x, dim_x) lower-triangular Cholesky factor of the residual's own process noise Q;
            h_next, (B, hidden_dim) updated hidden state
        """
        # When track is born create zero tensor for h
        B = x.shape[0]
        if h is None:
            h = torch.zeros(B, self.hidden_dim, dtype=x.dtype, device=x.device)

        # Create input vector
        dt_tensor = torch.full((B, 1), dt, dtype=x.dtype, device=x.device)
        gru_input = torch.concat((x, dt_tensor), dim=-1)

        # Run GRU cell to get h_next
        h_next = self.gru_cell(gru_input, h)

        # Compute delta
        delta = self.delta_head(h_next)

        # Compute L
        L = self.covariance_head(h_next)

        return (delta, L, h_next)



