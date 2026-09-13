import numpy as np
import torch

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







