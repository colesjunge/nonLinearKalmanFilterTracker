import torch
import torch.nn as nn


class DifferentiableKF(nn.Module):

    def __init__(self, F: torch.Tensor, H: torch.Tensor, Q: torch.Tensor) -> None:
        """ Differentiable Kalman Filter which implements the standard Kalman filter equations in a way that allows for backpropagation through the filter.
        In: F (dim_x, dim_x), state transition matrix
            H (dim_z, dim_x), measurement selection matrix
            Q (dim_x, dim_x), process noise covariance
        """
        
        super().__init__()
        # Stores the known, constant matrices as buffers (not learnable)
        self.register_buffer('F', F) # Motion model state transition matrix
        self.register_buffer('H', H) # Measurement model selection matrix
        self.register_buffer('Q', Q) # Process noise covariance


    def forward(self, z_seq: torch.Tensor, R_seq: torch.Tensor, dt: float, x0: torch.Tensor, P0: torch.Tensor, mask=None, eps: float = 1e-6, dynamics_fn=None, h0=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Forward pass through the differentiable Kalman filter
        If not None, dynamics_fn(x_t, dt, h_t) is called once per step to get (delta, L, h_next))
        These are used to update predictions using ResidualDynamicsNet
        In: z_seq (B, K, dim_z), sequence of raw measurements;
            R_seq (B, K, dim_z, dim_z), sequence of measurement noise covariances;
            dt (float), time step;
            x0 (B, dim_x), initial state;
            P0 (B, dim_x, dim_x), initial error covariance;
            mask (B, K) boolean (optional) mask for valid measurements (IMPLEMENT LATER)
            eps (float), small value to add to diagonal for numerical stability
            dynamics_fn (Callable[[Tensor(B,dim_x), float, Tensor(B,hidden_dim)|None], tuple[Tensor(B,dim_x), Tensor(B,dim_x,dim_x), Tensor(B,hidden_dim)]] | None, Same contract as ResidualDynamicsNet.forward();
            h0 (B, hidden_dim) | None, initial hidden state (only used when dynamics_fn not None)
        Out: x_seq (B, K, dim_x), sequence of estimated states;
             P_seq (B, K, dim_x, dim_x), sequence of estimated error covariances
             h_seq (B, K, hidden_dim), sequence of hidden states (all zeros when dynamix_fn is None)

        """

        # Set up storage for predicted states and covariances
        B, K, dim_z = z_seq.shape
        dim_x = x0.shape[1]

        # Store outputs at each timestep
        x_list = []
        P_list = []
        h_list = []

        # Initialize the state, covariance, and hidden state
        x_t = x0
        P_t = P0
        h_t = h0

        I_x = torch.eye(dim_x, dtype=P0.dtype, device=P0.device)
        I_z = torch.eye(dim_z, dtype=P0.dtype, device=P0.device)

        for t in range(K):

            # Predict step
            if dynamics_fn is not None:

                # Get learned residual and process noise factors
                delta, L, h_next = dynamics_fn(x_t, dt, h_t)

                # Construct learned process noise covariance
                Q_t = L @ L.transpose(-2, -1)

                # Incorporate learned residual into state prediction
                x_pred = x_t @ self.F.T + delta

                # Incorporate learned process noise into covariance prediction
                P_pred = self.F @ P_t @ self.F.T + Q_t

                # Carry hidden state forward
                h_t = h_next

                h_list.append(h_t)

            else:
                # Original fixed dynamics
                x_pred = x_t @ self.F.T
                P_pred = self.F @ P_t @ self.F.T + self.Q

            # Measurement and measurement covariance at time t
            z_t = z_seq[:, t, :]
            R_t = R_seq[:, t, :, :]

            # Compute innovation
            y_t = z_t - (x_pred @ self.H.T)

            # Compute innovation covariance and add jitter
            S_t = self.H @ P_pred @ self.H.T + R_t
            S_t += eps * I_z

            # Compute Kalman gain
            PH_T = P_pred @ self.H.T
            K_t = torch.linalg.solve(S_t, PH_T.transpose(-2, -1)).transpose(-2, -1)

            # Update state
            x_updated = x_pred + (K_t @ y_t.unsqueeze(-1)).squeeze(-1)

            # Update state and covariance (Joseph form symmetrized)
            I_KH = I_x - K_t @ self.H

            P_updated = (I_KH @ P_pred @ I_KH.transpose(-2, -1) + K_t @ R_t @ K_t.transpose(-2, -1))
            P_updated = 0.5 * (P_updated + P_updated.transpose(-2, -1))

            x_t = x_updated
            P_t = P_updated

            x_list.append(x_updated)
            P_list.append(P_updated)

        # Stack the list of states and covariances into tensors
        x_seq = torch.stack(x_list, dim=1)
        P_seq = torch.stack(P_list, dim=1)

        # Stack hidden states if applicable
        if dynamics_fn is not None:
            h_seq = torch.stack(h_list, dim=1)
        else:
            h_seq = torch.zeros(B, K, 0, dtype=x0.dtype, device=x0.device)

        return x_seq, P_seq, h_seq