import numpy as np

from tracking.filters.base import BayesFilter
from tracking.filters.kalman import KalmanFilter
from tracking.filters.utils import residual_with_angles
from fusion.filters.utils import gaussian_likelihood, gaussian_mixture_moments

class InteractingMultipleModel(BayesFilter):

    def __init__(self, filters: list[KalmanFilter], transition_matrix: np.ndarray, mode_probs0: np.ndarray) -> None:
        """ Stores markov mode switch probabilities
        In: filters, N already-constructed KalmanFilter instances (one per mode, each with its own motion model, same measurement model, same x0/P0 to start);
            transition_matrix (N, N), Markov mode-switch probabilities;
            mode_probs0 (N,), sums to 1
        """
        self.filters = filters
        self.transition_matrix = transition_matrix
        self.mode_probs = mode_probs0

        states = []
        for filter in filters:
            states.append((filter.x, filter.P))

        self.x, self.P = gaussian_mixture_moments(states, self.mode_probs)

    def predict(self, dt: float) -> None:
        """ Predict all modes after applying IMM state mixing
        In: dt, seconds since last update
        """
        N = len(self.filters)

        # Calculate predicted mode probabilities
        c = self.transition_matrix.T @ self.mode_probs

        # Calculate mixing probabilities
        mu = (self.mode_probs[:, None] * self.transition_matrix) / c[None, :]

        # Save all current filter states before mixing
        states = [(f.x.copy(), f.P.copy()) for f in self.filters]

        # Mix and predict each destination mode
        for j in range(N):

            # Combine previous mode estimates using column j of mu
            x_mixed, P_mixed = gaussian_mixture_moments(states, mu[:, j])

            # Initialize destination filter with its mixed estimate
            self.filters[j].x = x_mixed
            self.filters[j].P = P_mixed

            # Propagate using this mode's motion model
            self.filters[j].predict(dt)

        # Store predicted mode probabilities for the update step
        self.predicted_mode_probs = c

        # Combine the predicted filter estimates
        predicted_states = [(f.x, f.P) for f in self.filters]

        self.x, self.P = gaussian_mixture_moments(predicted_states, self.predicted_mode_probs)


    def update(self, z: np.ndarray) -> None:
        """ Update individual filters and IMM mode probabilities.
        In: z (dim_z,), measurement vector 
        """

        # Calculate the measurement likelihood for each mode
        likelihoods = []

        for f in self.filters:
            y, S = f.innovation(z)
            likelihoods.append(gaussian_likelihood(y, S))

        likelihoods = np.asarray(likelihoods)

        # Update each individual filter
        for f in self.filters:
            f.update(z)

        # Calculate posterior mode probabilities
        
        # Scale likelihoods 
        max_likelihood = np.max(likelihoods)

        scaled_likelihoods = likelihoods / max_likelihood

        posterior = scaled_likelihoods * self.predicted_mode_probs
        normalizer = np.sum(posterior)

        self.mode_probs = posterior / normalizer

        # Combine updated filter estimates
        updated_states = [(f.x, f.P) for f in self.filters]

        self.x, self.P = gaussian_mixture_moments(updated_states, self.mode_probs)

    def innovation(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the measurement innovation of the combined IMM estimate

        In: z (dim_z,), measurement vector;
        Out: y (dim_z,), Innovation vector;
            S (dim_z, dim_z), Innovation covariance matrix
        """

        # Use the first filter's measurement model
        f = self.filters[0]

        y = residual_with_angles(z, f.measurement.h(self.x), f.measurement.angle_indices)
       
        H = f.measurement.H(self.x)
        
        S = H @ self.P @ H.T + f.measurement.R()
        
        return (y, S)


