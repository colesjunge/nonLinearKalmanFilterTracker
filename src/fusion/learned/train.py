import numpy as np
import torch
import torch.nn as nn
from typing import Callable
from tqdm import tqdm

def run_model(model: nn.Module, batch: tuple) -> tuple[torch.Tensor | tuple, torch.Tensor]:
    """
    Helper function to run the model on a batch of data
    In: model (nn.Module), the network being trained;
        batch (tuple), a batch of data from the DataLoader, which can be either:
            (windows, targets) or (windows, targets, x0, P0)
    """

    if len(batch) == 2:
        windows, targets = batch
        output = model(windows)
        return output, targets
    
    elif len(batch) == 4:
        windows, targets, x0, P0 = batch
        output = model(windows, x0, P0)
        return output, targets

def train_epoch(model: nn.Module, loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, grad_clip_norm: float, loss_fn: Callable[[torch.Tensor | tuple, torch.Tensor], torch.Tensor]) -> float:
    """
    One full pass over the training data with weights updated
    In: model (nn.Module), the network being trained;
        loader (DataLoader), yields (window, target) batches, with each shaped (B, W, dim_z), (B, dim_pos);
        optimizer (torch.optim.Optimizer);
        grad_clip_norm (float), max gradient norm before clipping
        loss_fn (callaable), function to calculate loss
    Out: mean_loss, Mean loss (for training purposes)
    """

    model.train() # No effect here, but best practice

    losses = []

    pbar = tqdm(loader, desc="train_epoch", leave=False)
    for batch in pbar:

        output, targets = run_model(model, batch)

        optimizer.zero_grad() # Ensure gradient is overwritten
        
        loss = loss_fn(output, targets) # calculate loss against true positions
        loss.backward() # compute gradients
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm) # scale down gradients if they pass the clip
        optimizer.step() # apply update

        losses.append(loss.item())
        pbar.set_postfix(loss=loss.item())

    mean_loss = np.mean(losses)

    return mean_loss


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, loss_fn: Callable[[torch.Tensor | tuple, torch.Tensor], torch.Tensor]) -> float:
    """
    Evaluate model on validation data (no weight updates)
    In: model (nn.Module), the network being trained;
        loader (DataLoader), yields (window, target) batches, with each shaped (B, W, dim_z), (B, dim_pos)
        loss_fn (callaable), function to calculate loss

    Out: mean_loss, Mean loss (for validation purposes)

    """

    model.eval() # No effect here, but best practice

    losses = []
    with torch.no_grad(): # ensure no gradient tracking occurs (no computation wasted on building graph as no backward pass)
        eval_bar = tqdm(loader, desc="eval", leave=False)
        for batch in eval_bar:
            output, targets = run_model(model, batch)
            loss = loss_fn(output, targets) # calculate loss against true positions
            losses.append(loss.item())
            eval_bar.set_postfix(loss=loss)

    mean_loss = np.mean(losses)

    return mean_loss



def train(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, config: dict, loss_fn: Callable[[torch.Tensor | tuple, torch.Tensor], torch.Tensor], on_epoch_start: Callable[[], None] | None = None) -> dict:
    """
    Combines train_epoch and evaluate
    In: model (nn.Module), the network being trained;
        train_loader (DataLoader), yields (window, target) batches for training, with each shaped (B, W, dim_z), (B, dim_pos)
        val_loader (DataLoader), yields (window, target) batches for validation, with each shaped (B, W, dim_z), (B, dim_pos)
        config (dict), configuration variables
        loss_fn (callaable), function to calculate loss
        on_epoch_start (callable), function to call at the start of each epoch

    Out: history (dict), train_loss and val_loss for each epoch

    """

    history = {'train_loss': [], 'val_loss': []}
    
    # Create optimizer (Adaptive moment estimation use here)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    epoch_bar = tqdm(range(config['training']['epochs']), desc="train", leave=True)
    for _ in epoch_bar:
        if on_epoch_start:
            on_epoch_start()
        train_loss = train_epoch(model, train_loader, optimizer, config['training']['grad_clip_norm'], loss_fn)
        val_loss = evaluate(model, val_loader, loss_fn)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        epoch_bar.set_postfix(train_loss=train_loss, val_loss=val_loss)

    return history

    

    