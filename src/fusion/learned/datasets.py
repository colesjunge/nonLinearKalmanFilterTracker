import numpy as np
import torch

from fusion.learned.covariance import RolloutCovarianceModel

def compute_normalization_stats(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ Computes mean/std across all training windows per channel to standardize inputs
    In:  windows (n_traj, n_windows, window, dim_z), training trajectories only
    Out: mean (dim_z,), std (dim_z,)
    """
    mean = np.mean(windows, axis=(0,1,2))
    std = np.std(windows, axis=(0,1,2))

    return (mean, std)


class DenoiserDataset(torch.utils.data.Dataset):
    def __init__(self, windows: np.ndarray, targets: np.ndarray, mean: np.ndarray, std: np.ndarray):
        """
        Initialize DenoiserDataset
        in: windows/targets already restricted to one split (train or val) and already flattened
        to (n_samples, window, dim_z) and (n_samples, dim_pos)
        """
        self.windows = windows
        self.targets = targets 
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        """
        Return the amount of windows (samples)
        """
        return len(self.windows)
        
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return normalized window as well as target position
        In: idx int, index for given window/target
        Out: normalized window (window, dim_z), target position (dim_pos,), both as torch tensors
        """

        normalized_window = (self.windows[idx] - self.mean) / self.std
        target_position = self.targets[idx]

        normalized_window = torch.from_numpy(normalized_window).float()
        target_position = torch.from_numpy(target_position).float()


        return (normalized_window, target_position)

def load_denoiser_dataset(path, val_fraction: float, seed: int) -> tuple[DenoiserDataset, DenoiserDataset, dict]:
    """
    Load .npz dataset and split trajectories into train/val splits
    Flatten each split down to samples
    Compute normalization stats from train split only which are outputted
    In: path, path to .npz dataset; val_fraction, fraction that will be used for validation; seed, for rng
    Out: train_dataset, val_dataset, stats_dict {mean, std}
    """

    # Load data and get windows/targets
    data = np.load(path)
    windows = data['windows']
    targets = data['targets']

    # Permute and generate train/val indicies
    n_traj = windows.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_traj)
    n_val = int(n_traj * val_fraction)

    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_windows = windows[train_indices]
    train_targets = targets[train_indices]
    val_windows = windows[val_indices]
    val_targets = targets[val_indices]

    # Get mean and std prior to flattening
    train_mean, train_std = compute_normalization_stats(train_windows)

    # Flatten splits
    flattened_train_windows = train_windows.reshape(-1, train_windows.shape[2], train_windows.shape[3])
    flattened_train_targets = train_targets.reshape(-1, train_targets.shape[2])
    flattened_val_windows = val_windows.reshape(-1, val_windows.shape[2], val_windows.shape[3])
    flattened_val_targets = val_targets.reshape(-1, val_targets.shape[2])

    train_dataset = DenoiserDataset(flattened_train_windows, flattened_train_targets, train_mean, train_std)
    val_dataset = DenoiserDataset(flattened_val_windows, flattened_val_targets, train_mean, train_std) # Uses same train mean/std 

    stats_dict = {'mean': train_mean, 'std': train_std}

    return (train_dataset, val_dataset, stats_dict)


class RolloutWindowDataset(torch.utils.data.Dataset):
    def __init__(self, windows: np.ndarray, targets: np.ndarray, mean: np.ndarray, std: np.ndarray, K: int):
        """
        Initialize RolloutWindowDataset
        Computes (traj_idx, start_idx) from idx using integer divsion and modulo
        with each trajectory contributing (n_windows_per_traj - K + 1) possible starting indices for a rollout of length K
        This allows for the preservation of temporal structure
        in: windows (n_traj, n_windows_per_traj, window, dim_z), 
            targets (n_traj, n_windows_per_traj, dim_pos),
            mean (dim_z,),
            std (dim_z,),
            K (int) rollout length
        
        """
        self.windows = windows
        self.targets = targets
        self.mean = mean
        self.std = std
        self.K = K

        self.cache_x = None
        self.cache_P = None

    def __len__(self) -> int:
        """
        Return the amount of (traj_idx, start_idx) pairs for rollouts of length K
        """
        n_traj, n_windows_per_traj, _, _ = self.windows.shape
        return n_traj * (n_windows_per_traj - self.K + 1)
        
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return a samples z_window_seq and target_seq
        In: idx (int), index for given window/target
        Out: z_window_seq (K, W, dim_z), 
            target_seq (K, dim_pos), 
            x0 (dim_x,), 
            P0 (dim_x, dim_x)
        """
        
        n_traj, n_windows_per_traj, _, _ = self.windows.shape

        starts_per_traj = n_windows_per_traj - self.K + 1

        # Integer division finds how many full trajectories have been passed
        traj_idx = idx // starts_per_traj
        # Modulo finds the starting index within said trajectory
        start_idx = idx % starts_per_traj

        # Slice out the K-length sequence of windows and targets
        z_window_seq = self.windows[traj_idx, start_idx:start_idx+self.K]
        target_seq = self.targets[traj_idx, start_idx:start_idx+self.K]

        # Normalize the z_window_seq
        normalized_z_window_seq = (z_window_seq - self.mean) / self.std

        normalized_z_window_seq = torch.from_numpy(normalized_z_window_seq).float()
        target_seq = torch.from_numpy(target_seq).float()

        x0 = self.cache_x[traj_idx, start_idx]
        P0 = self.cache_P[traj_idx, start_idx]

        return (normalized_z_window_seq, target_seq, x0, P0)

def load_rollout_window_dataset(path, K: int, val_fraction: float, seed: int, mean: np.ndarray, std: np.ndarray) -> tuple[RolloutWindowDataset, RolloutWindowDataset, dict]:
    """
    Load .npz dataset and split trajectories into train/val splits
    Flatten each split down to samples
    In: path, path to .npz dataset;
        K, rollout length
        val_fraction, fraction that will be used for validation;
        seed, for rng; 
        mean, std, normalization stats from denoiser dataset
    Out: train_dataset, val_dataset, stats_dict {mean, std}
    """

    # Load data and get windows/targets
    data = np.load(path)
    windows = data['windows']
    targets = data['targets']

    # Permute and generate train/val indicies
    n_traj = windows.shape[0]
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_traj)
    n_val = int(n_traj * val_fraction)

    train_indices = indices[n_val:]
    val_indices = indices[:n_val]

    train_windows = windows[train_indices]
    train_targets = targets[train_indices]
    val_windows = windows[val_indices]
    val_targets = targets[val_indices]


    train_dataset = RolloutWindowDataset(train_windows, train_targets, mean, std, K)
    val_dataset = RolloutWindowDataset(val_windows, val_targets, mean, std, K) # Uses same train mean/std 

    stats_dict = {'mean': mean, 'std': std}

    return (train_dataset, val_dataset, stats_dict)


def compute_warm_states(model: RolloutCovarianceModel, windows: np.ndarray, mean: np.ndarray, std: np.ndarray, x0_init: torch.Tensor, P0_init: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute warm states for the rollout covariance model
    In: model, rollout covariance model
        windows (n_traj, n_windows_per_traj, window, dim_z), 
        mean (dim_z,), std (dim_z,), normalization stats from denoiser dataset
        x0_init (n_traj, dim_x), initial state estimate
        P0_init (n_traj, dim_x, dim_x), initial error covariance
    Out: all_x (n_traj, dim_x, n_windows_per_traj+1), all_P (n_traj, dim_x, dim_x, n_windows_per_traj+1), warm states for each trajectory
    """

    # Normalize the windows
    normalized_windows = (windows - mean) / std
    normalized_windows = torch.from_numpy(normalized_windows).float()

    # Compute warm states for each trajectory
    with torch.no_grad():
        x_filtered, P_filtered = model.forward(normalized_windows, x0_init, P0_init)
        all_x = torch.cat([x0_init.unsqueeze(1), x_filtered], dim=1)
        all_P = torch.cat([P0_init.unsqueeze(1), P_filtered], dim=1)

    return all_x, all_P






