import torch
import torch.nn as nn
from fusion.learned.denoiser import Denoiser


def save_checkpoint(path, model: nn.Module, normalization_stats: dict, config: dict) -> None:
    """ Build dictionary with this checkpoints learned weights, normalization stats, and config data
    In: model (nn.Module), the network being trained;
        normalization_stats (dict) {mean, std}
        config (dict), configuration variables
    Out: None, saves checkpoint_dict to path
    """

    checkpoint_dict = {'model_state_dict': model.state_dict(), 'normalization_stats': normalization_stats, 'config': config}
    torch.save(checkpoint_dict, path)

def load_checkpoint(path) -> tuple[nn.Module, dict, dict]:
    """
    Load checkpoint including model with weights, normalization stats, and config variables
    In: path, path to checkpoint_dict
    Out: model (nn.Module), the network being trained;
        normalization_stats (dict) {mean, std}
        config (dict), configuration variables
    """

    checkpoint = torch.load(path, weights_only=False) # Loading more than just the weights (config and norm stats)
    config = checkpoint['config']

    model = Denoiser(dim_z=config['model']['dim_z'],
                        dim_pos=config['model']['dim_pos'], 
                        hidden_channels=config['model']['hidden_channels'], 
                        n_layers=config['model']['n_layers'], 
                        kernel_size=config['model']['kernel_size'])

    model.load_state_dict(checkpoint['model_state_dict'])

    return (model, checkpoint['normalization_stats'], checkpoint['config'])