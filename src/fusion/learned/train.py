import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, grad_clip_norm: float) -> float:
    """
    One full pass over the training data with weights updated
    In: model (nn.Module), the network being trained;
        loader (DataLoader), yields (window, target) batches, with each shaped (B, W, dim_z), (B, dim_pos);
        optimizer (torch.optim.Optimizer);
        grad_clip_norm (float), max gradient norm before clipping
    Out: mean_loss, Mean loss (for training purposes)
    """

    model.train() # No effect here, but best practice

    losses = []

    for windows, targets in loader:
        optimizer.zero_grad() # Ensure gradient is overwritten

        predictions = model(windows) # Forward pass
        loss = F.mse_loss(predictions, targets) # calculate mse loss against true positions
        loss.backward() # compute gradients
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm) # scale down gradients if they pass the clip
        optimizer.step() # apply update

        losses.append(loss.item())

    mean_loss = np.mean(losses)

    return mean_loss


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader) -> float:
    """
    Evaluate model on validation data (no weight updates)
    In: model (nn.Module), the network being trained;
        loader (DataLoader), yields (window, target) batches, with each shaped (B, W, dim_z), (B, dim_pos)
    Out: mean_loss, Mean loss (for validation purposes)

    """

    model.eval() # No effect here, but best practice

    losses = []
    with torch.no_grad(): # ensure no gradient tracking occurs (no computation wasted on building graph as no backward pass)
        for windows, targets in loader:
            predictions = model(windows) # Forward pass
            loss = F.mse_loss(predictions, targets) # calculate mse loss against true positions
            losses.append(loss.item())

    mean_loss = np.mean(losses)

    return mean_loss



def train(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, config: dict) -> dict:
    """
    Combines train_epoch and evaluate
    In: model (nn.Module), the network being trained;
        train_loader (DataLoader), yields (window, target) batches for training, with each shaped (B, W, dim_z), (B, dim_pos)
        val_loader (DataLoader), yields (window, target) batches for validation, with each shaped (B, W, dim_z), (B, dim_pos)
        config (dict), configuration variables
    Out: history (dict), train_loss and val_loss for each epoch

    """

    history = {'train_loss': [], 'val_loss': []}
    
    # Create optimizer (Adaptive moment estimation use here)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    for _ in range(config['training']['epochs']):
        train_loss = train_epoch(model, train_loader, optimizer, config['training']['grad_clip_norm'])
        val_loss = evaluate(model, val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

    return history

    

    