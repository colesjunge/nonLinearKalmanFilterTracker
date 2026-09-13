import torch
import torch.nn as nn
import torch.nn.functional as F

class Denoiser(nn.Module):
    """
    Learned measurement denoiser that maps a causal window of raw (contaminated noise) radar measurements to a single denoised Cartesian position estimate
    Uses a stack of causal dilated 1-D convolutions (the receptive field grows exponentially with depth)
    Padding is manually applied in the forward(), on the left side only, to guarantee that the estimate never depends on data beyond it
    """

    def __init__(self, dim_z: int, dim_pos: int, hidden_channels: int, n_layers: int, kernel_size: int):

        super().__init__()
        self.dim_z = dim_z
        self.dim_pos = dim_pos
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers
        self.kernel_size = kernel_size

        self.convs = nn.ModuleList()

        for i in range(n_layers):
            in_channels = dim_z if i == 0 else hidden_channels
            dilation = 2 ** i

            self.convs.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=0 # Padding will be done in forward() do reduce causality risks
                )
            )

        self.output_layer = nn.Linear(hidden_channels, dim_pos)

    def _conv_features(self, z_window: torch.Tensor) -> torch.Tensor:
        """
        Helper function to forward to have access to intermediate features
        In:  z_window (B, W, dim_z)
        Out: x (B, dim_z, W)
        """
        # z_window is (B, W, dim_z), but Conv1d expects (B, dim_z, W)
        x = z_window.transpose(1, 2)

        # Padding ensures that the estimate at time j can only use everything up to j, but nothing after (casual)
        for conv in self.convs:
            pad_amount = (self.kernel_size - 1) * conv.dilation[0]

            x = F.pad(x, (pad_amount, 0))
            x = conv(x)
            x = F.relu(x)

        return x

    def forward(self, z_window: torch.Tensor) -> torch.Tensor:
        """ Map causal window of raw measurements to a single denoised position estimate (final)
        In:  z_window (B, W, dim_z)
        Out: (B, dim_pos)
        """

        x = self._conv_features(z_window)

        # Keep last timestep's feature vector
        x = x[:, :, -1]
        # x is (B, hidden_channels, W), but only need (B, dim_pos)
        x = self.output_layer(x)

        return x
    