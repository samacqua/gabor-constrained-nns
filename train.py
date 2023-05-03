"""Trains the models."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
import os


def train_many(models: list[nn.Module], optimizers: list[optim.Optimizer], model_names: list[str], 
               model_infos: list[dict], dataloader: torch.utils.data.DataLoader, save_dir: str, device: str = 'cpu', 
               starting_epoch: int = 0, n_epochs: int = 40, criterion: torch.nn.Module = nn.CrossEntropyLoss()):
    """Trains multiple models."""
    
    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tensorboard"))
    for model in models:
        model.train()

    for epoch in range(starting_epoch, n_epochs):

        print(f"=======\nEpoch {epoch + 1}/{n_epochs}\n=======")

        losses = {name: [] for name in model_names}
        correct = {name: [] for name in model_names}
        n_total = []
    
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):

            batch_idx = epoch * len(dataloader) + i

            for model, optimizer, name in zip(models, optimizers, model_names):

                # Forward + backward + optimize.
                optimizer.zero_grad()
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                # Calculate stats + log.
                pred = outputs.max(1, keepdim=True)[1].to("cpu")
                correct[name].append(pred.eq(labels.view_as(pred)).sum().item())
                losses[name].append(loss.item())

                writer.add_scalars("Loss/train", {name: loss.item()}, batch_idx)
                writer.add_scalars(
                    "Accuracy/train",
                    {name: correct[name][-1] / len(labels)},
                    batch_idx,
                )

        # Save the model + optimizer state.
        for model, optimizer, name, model_info in zip(models, optimizers, model_names, model_infos):
            model_save_dir = os.path.join(save_dir, "models", name)
            os.makedirs(model_save_dir, exist_ok=True)
            torch.save({
                **{'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses[name],
                'correct': correct[name],
                'n_total': n_total},
                **model_info,
            }, os.path.join(model_save_dir, f"epoch_{epoch}.pth"))


def train(model: torch.nn.Module, train_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, 
          criterion: torch.nn.Module, device: torch.device, starting_epoch: int = 0, n_epochs: int = 10, save_dir: str = None, 
          model_name: str = None, model_info: dict = None):
    """Trains a model."""
    train_many([model], [optimizer], [model_name], [model_info], dataloader=train_loader, save_dir=save_dir, device=device,
               starting_epoch=starting_epoch, n_epoch=n_epochs, criterion=criterion)


if __name__ == '__main__':
    pass
