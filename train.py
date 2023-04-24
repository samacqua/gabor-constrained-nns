"""Trains the models."""

import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import yaml


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
          criterion: torch.nn.Module, device: torch.device, epochs: int = 1, log_dir: str = None):
    """Trains a model."""
    model.train()
    writer = SummaryWriter(log_dir=log_dir)

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # if batch_idx == 50:
        #     break

        writer.add_scalar('train/loss', loss.item(), batch_idx)


if __name__ == '__main__':
    pass
