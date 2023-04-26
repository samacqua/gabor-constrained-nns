"""Trains the models."""

import torch
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import os


def count_parameters(model, trainable_only=True):
    n_by_layer = [p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)]
    return sum(n_by_layer), n_by_layer


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
          criterion: torch.nn.Module, device: torch.device, epochs: int = 1, log_dir: str = None, 
          save_every: int = None, model_save_dir: str = None, model_suffix: str = None):
    """Trains a model."""

    # Check that arguments are valid.
    if save_every is not None:
        assert model_save_dir is not None
        assert model_suffix is not None

    model.train()
    model.to(device)
    writer = SummaryWriter(log_dir=log_dir)

    for epoch in range(epochs):
        for i, (data, target) in enumerate(tqdm(train_loader)):

            batch_idx = i + epoch * len(train_loader)

            # Run training step.
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # # Print the trainable parameter counts to ensure they're frozen / configured correctly.
            # if batch_idx == 0:
            #     print(model.conv1)
            #     print(count_parameters(model))
            #     print(count_parameters(model, False))

            # Calc batch accuracy.
            pred = output.argmax(dim=1, keepdim=True)
            acc = pred.eq(target.view_as(pred)).sum().item() / len(data)

            # Log loss + accuracy + save model.
            writer.add_scalar('train/loss', loss.item(), batch_idx)
            writer.add_scalar('train/accuracy', acc, batch_idx)
            if save_every and batch_idx % save_every == 0:
                save_path = os.path.join(model_save_dir, f"model_{model_suffix}_{batch_idx}.pt")
                torch.save(model.state_dict(), save_path)

    if save_every:
        save_path = os.path.join(model_save_dir, f"model_{model_suffix}.pt")
        torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    pass
