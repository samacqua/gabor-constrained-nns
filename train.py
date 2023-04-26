"""Trains the models."""

import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import os


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
          criterion: torch.nn.Module, device: torch.device, epochs: int = 1, log_dir: str = None, 
          save_every: int = None, model_save_dir: str = None, model_suffix: str = None):
    """Trains a model."""

    # Check that arguments are valid.
    if save_every is not None:
        assert model_save_dir is not None
        assert model_suffix is not None

    model.train()
    writer = SummaryWriter(log_dir=log_dir)

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):

        # Run training step.
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # Log.
        writer.add_scalar('train/loss', loss.item(), batch_idx)
        if save_every and batch_idx % save_every == 0:
            save_path = os.path.join(model_save_dir, f"model_{model_suffix}_{batch_idx}.pt")
            torch.save(model.state_dict(), save_path)

    if save_every:
        save_path = os.path.join(model_save_dir, f"model_{model_suffix}.pt")
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    pass
