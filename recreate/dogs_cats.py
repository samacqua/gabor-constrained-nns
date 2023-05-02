"""File to recreate the results from the original paper on the Dogs v. Cats dataset.

The original paper is: https://arxiv.org/abs/1904.13204
Using the dog vs. cat: https://www.kaggle.com/competitions/dogs-vs-cats/data
Based on https://github.com/iKintosh/GaborNet/blob/master/sanity_check/run_sanity_check.py to keep it as similar to the
original paper as possible.
"""

import json
import os
from tqdm import tqdm
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from models import GaborConv2d
from .dataset import DogsCatsDataset

from torch.utils.tensorboard import SummaryWriter


class DogCatNNSanity(nn.Module):
    """From https://github.com/iKintosh/GaborNet/blob/master/sanity_check/run_sanity_check.py.
    
    This is not the network used in the original paper, but the authors do not have a public implementation of their
    tests, so this serves as a good starting point of something the authors have implemented.
    """

    def __init__(self, is_gabornet: bool = False):
        super(DogCatNNSanity, self).__init__()
        if is_gabornet:
            self.g1 = GaborConv2d(3, 32, kernel_size=(15, 15), stride=1)
        else:
            self.g1 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=1)

        self.c1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)
        self.c2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.g1(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        x = F.max_pool2d(F.leaky_relu(self.c1(x)), kernel_size=2)
        x = F.max_pool2d(F.leaky_relu(self.c2(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        x = x.view(-1, 128 * 7 * 7)
        x = F.leaky_relu(self.fc1(x))
        x = nn.Dropout()(x)
        x = self.fc3(x)
        return x


class DogCatNet(nn.Module):
    """Based on figure in original paper.
    
    The original paper says that Dropout is applied, but it is not clear where; we assume that Dropout is applied after
    every non-final layer. The figure specifies exact shapes, but the shapes are 
        - for GaborNet: off-by-one in the first layer (N, 32, 121, 121) instead of the reported (N, 32, 120, 120). This
            converges to the right shapes afterwards.
        - for CNN: fairly different shapes throughout the architectures. This leads to needing a different number of
            parameters for the first linear layer. The authors report that, even though the first layer has a different
            kernel size than in GaborNet, all the other layers are the same parameters / shapes.
     
    It isn't clear why the authors chose to compare a 15x15 Gabor kernel to a 5x5 CNN kernel. It seems like a fairer
    comparison would be to have kernels with either the same number of parameters (2x2 CNN kernel) or the same size 
    (15x15 CNN kernel).
    """

    def __init__(self, is_gabornet: bool = False):
        super(DogCatNet, self).__init__()

        if is_gabornet:
            self.g1 = GaborConv2d(in_channels=3, out_channels=32, kernel_size=(15, 15), stride=1)
        else:
            self.g1 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=1)

        self.c1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1)
        self.c2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1)
        self.c4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1)

        if is_gabornet:
            self.fc1 = nn.Linear(3200, 128)
        else:
            self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

        self.is_gabornet = is_gabornet

    def forward(self, x):

        N = x.shape[0]
        assert x.shape == (N, 3, 256, 256)

        # Should be (N, 32, 120, 120) according to paper.
        x1 = F.max_pool2d(F.relu(self.g1(x)), kernel_size=2)
        x1 = nn.Dropout2d()(x1)
        # assert x1.shape == (N, 32, 120, 120)

        # Should be (N, 64, 59, 59) according to paper.
        x2 = F.max_pool2d(F.relu(self.c1(x1)), kernel_size=2)
        x2 = nn.Dropout2d()(x2)
        # assert x2.shape == (N, 64, 59, 59)

        # Should be (N, 128, 28, 28) according to paper.
        x3 = F.max_pool2d(F.relu(self.c2(x2)), kernel_size=2)
        x3 = nn.Dropout2d()(x3)
        # assert x3.shape == (N, 128, 28, 28)

        # Should be (N, 128, 13, 13) according to paper.
        x = F.max_pool2d(F.relu(self.c3(x3)), kernel_size=2)
        x = nn.Dropout2d()(x)
        # assert x.shape == (N, 128, 13, 13)

        # Should be (N, 128, 5, 5) according to paper.
        x = F.max_pool2d(F.relu(self.c4(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        # assert x.shape == (N, 128, 5, 5)

        # Since the shapes don't match between the 5x5 CNN and 15x15 GaborNet, we need to reshape differently.
        if self.is_gabornet:
            x = x.view(-1, 3200)
        else:
            x = x.view(-1, 128 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = nn.Dropout()(x)
        assert x.shape == (N, 128)

        x = F.relu(self.fc2(x))
        x = nn.Dropout()(x)
        assert x.shape == (N, 128)

        x = self.fc3(x)
        assert x.shape == (N, 2)

        return x


def load_net(fpath: str, model: torch.nn.Module, optimizer: optim.Optimizer = None
             ) -> tuple[nn.Module, optim.Optimizer, int]:
    """Loads a model from a file, with the optimizer and epoch."""

    # Load the model.
    if model.is_gabornet:
        # Needed to avoid some memory issues.
        model.g1.x = Parameter(model.g1.x.contiguous())
        model.g1.y = Parameter(model.g1.y.contiguous())
        model.g1.x_grid = Parameter(model.g1.x_grid.contiguous())
        model.g1.y_grid = Parameter(model.g1.y_grid.contiguous())

    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load the optimizer.
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, checkpoint["epoch"]


def load_dataset(dataset_dir: str = "data/dogs-vs-cats/"):
    """Loads the cats v. dogs dataset."""

    # Noramlize the data.
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = DogsCatsDataset(root_dir=os.path.join(dataset_dir, "train"), transform=transform)
    test_set = DogsCatsDataset(root_dir=os.path.join(dataset_dir, "test1"), transform=transform)

    return train_set, test_set


def main():

    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--dataset_dir", type=str, default="data/dogs-vs-cats/", help="Path to dataset.")
    parser.add_argument("--model", type=str, default="DogCatNet")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint.")
    args = parser.parse_args()

    rand_seed = args.seed if args.seed is not None else np.random.randint(0, 10000)
    save_dir = os.path.join("recreate/out/", f"seed_{rand_seed}/")

    if os.path.exists(save_dir) and not args.resume:    # Make sure we are not overwriting on accident.
        should_continue = input(f"Overwriting existing files ({save_dir}). Continue? [y/n] ")
        if should_continue.lower() != "y":
            exit(0)

    if args.model == "DogCatNet":
        net_arch = DogCatNet
    elif args.model == "DogCatNNSanity":
        net_arch = DogCatNNSanity
    else:
        raise ValueError(f"Invalid model {args.model}.")

    cnn_load_path, gabor_load_path = None, None
    if args.resume:
        # Find the latest version of the model. fnames of form: "{model_name}_epoch_{epoch}.pth"
        fnames = [fname for fname in os.listdir(save_dir) if fname.endswith(".pth")]
        cnn_epoch = max([int(fname.split("epoch_")[1].split(".pth")[0]) for fname in fnames if "cnn" in fname])
        gabor_epoch = max([int(fname.split("epoch_")[1].split(".pth")[0]) for fname in fnames if "gabor" in fname])
        assert cnn_epoch == gabor_epoch, "CNN and GaborNet epochs do not match."
        cnn_load_path = os.path.join(save_dir, f"cnn_epoch_{cnn_epoch}.pth")
        gabor_load_path = os.path.join(save_dir, f"gabornet_epoch_{gabor_epoch}.pth")

    # Hyperparameters from paper.
    BATCH_SIZE = 64
    OPT = optim.AdamW
    N_EPOCHS = 40
    LR = 0.001
    BETAS = (0.9, 0.999)

    # Load the dataset.
    torch.manual_seed(rand_seed)
    train_set, test_set = load_dataset(args.dataset_dir)

    # Paper says they use 30% of the trainset as the validation set.
    # Just going to do that via code.
    train_set, _ = torch.utils.data.random_split(
        train_set, [int(len(train_set) * 0.7), int(len(train_set) * 0.3)]
    )
    train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the models + optimizers.
    gabornet = net_arch(is_gabornet=True).to(device)
    cnn = net_arch(is_gabornet=False).to(device)
    gabornet_optimizer = OPT(gabornet.parameters(), lr=LR, betas=BETAS)
    cnn_optimizer = OPT(cnn.parameters(), lr=LR, betas=BETAS)

    starting_epoch = 0
    if cnn_load_path and gabor_load_path:
        gabornet, gabornet_optimizer, last_epoch_gabor = load_net(gabor_load_path, gabornet, gabornet_optimizer)
        starting_epoch = last_epoch_gabor + 1

        cnn, cnn_optimizer, last_epoch_cnn = load_net(cnn_load_path, cnn, cnn_optimizer)
        assert starting_epoch == last_epoch_cnn + 1

    # Train the models.
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=save_dir)

    gabornet.train()
    cnn.train()

    for epoch in range(starting_epoch, N_EPOCHS):

        print(f"=======\nEpoch {epoch + 1}/{N_EPOCHS}\n=======")

        gab_loss = []
        cnn_loss = []

        gabornet_correct = []
        cnn_correct = []
        n_total = []
    
        for i, data in enumerate(tqdm(train)):

            batch_idx = epoch * len(train) + i
            inputs, labels = data["image"], data["target"]

            # === GABORNET ===

            gabornet_optimizer.zero_grad()
            outputs = gabornet(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            gabornet_optimizer.step()

            # Calculate stats + log.
            pred = outputs.max(1, keepdim=True)[1].to("cpu")
            gabornet_correct.append(pred.eq(labels.view_as(pred)).sum().item())
            gab_loss.append(loss.item())

            writer.add_scalars("Loss/train", {'gabornet': loss.item()}, batch_idx)
            writer.add_scalars(
                "Accuracy/train",
                {'gabornet': gabornet_correct[-1] / len(labels)},
                batch_idx,
            )

            # === CNN ===

            cnn_optimizer.zero_grad()
            outputs = cnn(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            cnn_optimizer.step()

            # Calculate stats + log.
            pred = outputs.max(1, keepdim=True)[1].to("cpu")
            cnn_correct.append(pred.eq(labels.view_as(pred)).sum().item())
            cnn_loss.append(loss.item())

            writer.add_scalars("Loss/train", {'cnn': loss.item()}, batch_idx)
            writer.add_scalars(
                "Accuracy/train",
                {'cnn': cnn_correct[-1] / len(labels)},
                batch_idx,
            )

        # Save the model + optimizer state for gabornet.
        torch.save({
            'epoch': epoch,
            'model_state_dict': gabornet.state_dict(),
            'optimizer_state_dict': gabornet_optimizer.state_dict(),
            'loss': gab_loss,
            'correct': gabornet_correct,
            'n_total': n_total,
        }, os.path.join(save_dir, f"gabornet_epoch_{epoch}.pth"))

        # Save the model + optimizer state for cnn.
        torch.save({
            'epoch': epoch,
            'model_state_dict': cnn.state_dict(),
            'optimizer_state_dict': cnn_optimizer.state_dict(),
            'loss': cnn_loss,
            'correct': cnn_correct,
            'n_total': n_total,
        }, os.path.join(save_dir, f"cnn_epoch_{epoch}.pth"))

    print("Finished Training")


if __name__ == "__main__":
    main()
