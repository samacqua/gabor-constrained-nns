"""File to recreate the results from the original paper on the Dogs v. Cats dataset.

The original paper is: https://arxiv.org/abs/1904.13204
Using the dog vs. cat: https://www.kaggle.com/competitions/dogs-vs-cats/data
Based on https://github.com/iKintosh/GaborNet/blob/master/sanity_check/run_sanity_check.py to keep it as similar to the
original paper as possible.
"""

from typing import Type
import os
from tqdm import tqdm
import argparse
import numpy as np
import json

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from train import train_many
from gabor_layer import GaborConv2dPip, GaborConv2dGithub, GaborConv2dGithubUpdate
from .dataset import DogsCatsDataset


class DogCatNNSanity(nn.Module):
    """From https://github.com/iKintosh/GaborNet/blob/master/sanity_check/run_sanity_check.py.
    
    This is not the network used in the original paper, but the authors do not have a public implementation of their
    tests, so this serves as a good starting point of something the authors have implemented.
    """

    def __init__(self, is_gabornet: bool = False, kernel_size: tuple[int, int] = (15, 15), add_padding: bool = True, 
                 gabor_type: Type[nn.ModuleDict] = GaborConv2dPip, device="cpu", bias: bool = True):
        super(DogCatNNSanity, self).__init__()

        self.is_gabornet = is_gabornet
        if is_gabornet:
            self.g1 = gabor_type(3, 32, kernel_size=kernel_size, stride=1, device=device)
        else:
            self.g1 = nn.Conv2d(3, 32, kernel_size=kernel_size, stride=1, bias=bias)

        self.c1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)
        self.c2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2)
        self.fc1 = nn.Linear(128 * 7 * 7, 128)
        self.fc3 = nn.Linear(128, 2)

        self.dropout2d = nn.Dropout2d()

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.g1(x)), kernel_size=2)
        x = self.dropout2d(x)
        x = F.max_pool2d(F.leaky_relu(self.c1(x)), kernel_size=2)
        x = F.max_pool2d(F.leaky_relu(self.c2(x)), kernel_size=2)
        x = self.dropout2d(x)
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

    add_padding is a boolean that specifies whether to add padding before the first connected layer. Used to make sure 
    parameter counts are equal after the original layer. 
    """

    def __init__(self, is_gabornet: bool = False, kernel_size: tuple[int, int] = (15, 15), add_padding: bool = False, 
                 gabor_type: Type[nn.ModuleDict] = GaborConv2dPip, device="cpu", bias: bool = True):
        super(DogCatNet, self).__init__()

        self.is_gabornet = is_gabornet
        if is_gabornet:
            self.g1 = gabor_type(in_channels=3, out_channels=32, kernel_size=kernel_size, stride=1, device=device)
        else:
            self.g1 = nn.Conv2d(3, 32, kernel_size=kernel_size, stride=1, bias=bias)

        self.c1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=1)
        self.c2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1)
        self.c3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1)
        self.c4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=1)

        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

        self.dropout2d = nn.Dropout2d()
        self.dropout = nn.Dropout()

        self.is_gabornet = is_gabornet
        self.add_padding = add_padding

    def forward(self, x):

        N = x.shape[0]
        # assert x.shape == (N, 3, 256, 256)

        # Should be (N, 32, 120, 120) according to paper.
        x1 = F.max_pool2d(F.relu(self.g1(x)), kernel_size=2)
        x1 = self.dropout2d(x1)
        # assert x1.shape == (N, 32, 120, 120)

        # Should be (N, 64, 59, 59) according to paper.
        x2 = F.max_pool2d(F.relu(self.c1(x1)), kernel_size=2)
        x2 = self.dropout2d(x2)
        # assert x2.shape == (N, 64, 59, 59)

        # Should be (N, 128, 28, 28) according to paper.
        x3 = F.max_pool2d(F.relu(self.c2(x2)), kernel_size=2)
        x3 = self.dropout2d(x3)
        # assert x3.shape == (N, 128, 28, 28)

        # Should be (N, 128, 13, 13) according to paper.
        x = F.max_pool2d(F.relu(self.c3(x3)), kernel_size=2)
        x = self.dropout2d(x)
        # assert x.shape == (N, 128, 13, 13)

        # Should be (N, 128, 5, 5) according to paper.
        x = F.max_pool2d(F.relu(self.c4(x)), kernel_size=2, padding=int(self.add_padding))
        x = self.dropout2d(x)
        self._x_size = x.shape

        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        assert x.shape == (N, 128)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        assert x.shape == (N, 128)

        x = self.fc3(x)
        assert x.shape == (N, 2)

        return x


def determine_padding(model_arch: Type[torch.nn.Module], gabor_kernel: tuple[int, int], cnn_kernel: tuple[int, int]
                      ) -> tuple[bool, bool]:
    """Determines which model, if either, needs padding to have the same architecture post-filter."""

    if model_arch != DogCatNet:
        raise NotImplementedError("This function only works for the DogCatNet architecture.")

    # Run a random image of the correct shape through the model to calculate the size before the linear layers.
    fake_img = torch.randn(1, 3, 256, 256)

    gabor_model = model_arch(is_gabornet=True, kernel_size=gabor_kernel, add_padding=False)
    cnn_model = model_arch(is_gabornet=False, kernel_size=cnn_kernel, add_padding=False)
    _ = gabor_model(fake_img)
    _ = cnn_model(fake_img)

    # Determine if padding is needed.
    if gabor_model._x_size == cnn_model._x_size:
        return False, False
    elif gabor_model._x_size == (1, 128, 6, 6) and cnn_model._x_size == (1, 128, 5, 5):
        gabor_padding = False
        cnn_padding = True
    elif gabor_model._x_size == (1, 128, 5, 5) and cnn_model._x_size == (1, 128, 6, 6):
        gabor_padding = True
        cnn_padding = False
    else:
        raise NotImplementedError("Simple heuristic to match architectures doesn't work.")
    
    # Test that the padding works.
    gabor_model = model_arch(is_gabornet=True, kernel_size=gabor_kernel, add_padding=gabor_padding)
    cnn_model = model_arch(is_gabornet=False, kernel_size=cnn_kernel, add_padding=cnn_padding)
    _ = gabor_model(fake_img)
    _ = cnn_model(fake_img)
    assert gabor_model._x_size == cnn_model._x_size

    return gabor_padding, cnn_padding


def load_net(checkpoint, model: torch.nn.Module, optimizer: optim.Optimizer = None, strict: bool = True
             ) -> tuple[nn.Module, optim.Optimizer, int]:
    """Loads a model from a file, with the optimizer and epoch."""

    # Load the model.
    if model.is_gabornet and isinstance(model.g1, (GaborConv2dGithub, GaborConv2dGithubUpdate)):
        # Needed to avoid some memory issues.
        model.g1.x = Parameter(model.g1.x.contiguous())
        model.g1.y = Parameter(model.g1.y.contiguous())
        model.g1.x_grid = Parameter(model.g1.x_grid.contiguous())
        model.g1.y_grid = Parameter(model.g1.y_grid.contiguous())

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load the optimizer.
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, checkpoint["epoch"]


def load_dataset(dataset_dir: str = "data/dogs-vs-cats/", img_size: tuple[int, int] = (256, 256)):
    """Loads the cats v. dogs dataset."""

    # Noramlize the data.
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_set = DogsCatsDataset(root_dir=os.path.join(dataset_dir, "train"), transform=transform)
    test_set = DogsCatsDataset(root_dir=os.path.join(dataset_dir, "test1"), transform=transform)

    return train_set, test_set


def main(args):

    # If initializing the CNN with the GaborNet weights, the CNN should not have a bias.
    cnn_bias = not args.init_cnn_with_gabor

    rand_seed = args.seed if args.seed is not None else np.random.randint(0, 10000)
    save_dir = os.path.join("recreate/out/", (args.dir_name or f"seed_{rand_seed}/"))

    if os.path.exists(save_dir) and not args.resume:    # Make sure we are not overwriting on accident.
        should_continue = input(f"Overwriting existing files ({save_dir}). Continue? [y/n] ")
        if should_continue.lower() != "y":
            exit(0)
    os.makedirs(save_dir, exist_ok=True)

    net_arch = globals()[args.model]
    gabor_arch = globals()[args.gabor_type]

    # Load the latest checkpoint.
    cnn_checkpoint, gabor_checkpoint = None, None
    if args.resume:
        if not args.only_gabor:
            try:
                cnn_dir = os.path.join(save_dir, "models", "cnn")
                cnn_epoch = max([int(fname.split("epoch_")[1].split(".pth")[0]) for fname in os.listdir(cnn_dir)])
                cnn_checkpoint = torch.load(os.path.join(cnn_dir, f"epoch_{cnn_epoch}.pth"))
            except FileNotFoundError:
                cnn_checkpoint = None
                cnn_epoch = 0
        if not args.only_cnn:
            try:
                gabor_dir = os.path.join(save_dir, "models", "gabornet")
                gabor_epoch = max([int(fname.split("epoch_")[1].split(".pth")[0]) for fname in os.listdir(gabor_dir)])
                gabor_checkpoint = torch.load(os.path.join(gabor_dir, f"epoch_{gabor_epoch}.pth"))
            except FileNotFoundError:
                gabor_checkpoint = None
                gabor_epoch = 0
        if not args.only_gabor and not args.only_cnn:
            assert cnn_epoch == gabor_epoch, "CNN and GaborNet epochs from filenames do not match."

    # Save the arguments.
    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f)

    # Hyperparameters from paper.
    BATCH_SIZE = 64
    OPT = optim.AdamW
    N_EPOCHS = args.epochs
    LR = 0.001
    BETAS = (0.9, 0.999)

    # Load the dataset.
    torch.manual_seed(rand_seed)
    train_set, test_set = load_dataset(args.dataset_dir, (args.img_size, args.img_size))

    # Paper says they use 30% of the trainset as the validation set.
    # Just going to do that via code.
    N = int(len(train_set) * 0.7)
    # N = 128
    train_set, _ = torch.utils.data.random_split(
        train_set, [N, len(train_set) - N]
    )
    
    train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the models + optimizers.
    if args.padding:
        gabor_padding, cnn_padding = determine_padding(net_arch, args.gabor_kernel, args.cnn_kernel)
    else:
        gabor_padding, cnn_padding = False, False
    gabornet = net_arch(is_gabornet=True, kernel_size=args.gabor_kernel, add_padding=gabor_padding, 
                        gabor_type=gabor_arch, device=device).to(device)
    cnn = net_arch(is_gabornet=False, kernel_size=args.cnn_kernel, add_padding=cnn_padding, 
                   gabor_type=gabor_arch, bias=cnn_bias).to(device)

    # Initialize the CNN with a Gabor function.
    if args.init_cnn_with_gabor:
        assert cnn_padding == gabor_padding == False
        gabor_weight = gabornet.g1.calculate_weights()
        cnn.g1.weight.data = gabor_weight.clone().detach()
        assert cnn.g1.bias is None

    gabornet_optimizer = OPT(gabornet.parameters(), lr=LR, betas=BETAS)
    cnn_optimizer = OPT(cnn.parameters(), lr=LR, betas=BETAS)

    # Load the model + optimizer checkpoints.
    starting_epoch = 0
    if gabor_checkpoint:
        gabornet, gabornet_optimizer, last_epoch = load_net(gabor_checkpoint, gabornet, gabornet_optimizer)
        starting_epoch = last_epoch + 1
    if cnn_checkpoint:
        cnn, cnn_optimizer, last_epoch = load_net(cnn_checkpoint, cnn, cnn_optimizer)
        assert (starting_epoch == last_epoch + 1) or args.only_cnn, "CNN and GaborNet epochs do not match."
        starting_epoch = last_epoch + 1

    # Freeze the first layer.
    if args.freeze_cnn:
        for param in cnn.g1.parameters():
            param.requires_grad = False
    if args.freeze_gabor:
        for param in gabornet.g1.parameters():
            param.requires_grad = False

    # Train the models.
    models = [gabornet, cnn]
    optimizers = [gabornet_optimizer, cnn_optimizer]
    model_names = ["gabornet", "cnn"]
    model_infos = [
        {'kernel_size': args.gabor_kernel, 'add_padding': gabor_padding, "gabor_type": args.gabor_type},
        {'kernel_size': args.cnn_kernel, 'add_padding': cnn_padding, "gabor_type": None}
    ]

    # Only train one model.
    if args.only_cnn or args.only_gabor:
        index = int(args.only_cnn)

        models = [models[index]]
        optimizers = [optimizers[index]]
        model_names = [model_names[index]]
        model_infos = [model_infos[index]]

    train_many(models=models, optimizers=optimizers, model_names=model_names, model_infos=model_infos, 
               dataloader=train, save_dir=save_dir, device=device, starting_epoch=starting_epoch, n_epochs=N_EPOCHS)


if __name__ == "__main__":

    # Parse arguments.
    parser = argparse.ArgumentParser()

    # Experiment params.
    parser.add_argument("--seed", type=int, default=None, help="Random seed.")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint.")
    parser.add_argument("--dir_name", type=str, default=None, 
                        help="Name of directory to save in. Defaults to seed_{random seed}.")

    # Model params.
    parser.add_argument("--model", type=str, default="DogCatNet", choices=["DogCatNet", "DogCatNNSanity"])
    parser.add_argument("--gabor_kernel", type=int, default=15, help="Size of GaborNet kernel.")
    parser.add_argument("--cnn_kernel", type=int, default=5, help="Size of CNN kernel.")
    parser.add_argument("--padding", action="store_true", help="Use padding to even the paramters.")
    parser.add_argument("--gabor_type", type=str, default="GaborConv2dPip", help="Type of GaborNet to use.", 
                        choices=["GaborConv2dPip", "GaborConv2dGithub", "GaborConv2dGithubUpdate"])

    # Training params.
    parser.add_argument("--init_cnn_with_gabor", action="store_true", help="Initialize CNN with GaborNet weights.")
    parser.add_argument("--freeze_cnn", action="store_true", help="Freeze the first layer of the CNN.")
    parser.add_argument("--freeze_gabor", action="store_true", help="Freeze the first layer of the GaborNet.")
    parser.add_argument("--only_cnn", action="store_true", help="Flag to train only the CNN model.")
    parser.add_argument("--only_gabor", action="store_true", help="Flag to train only the GaborNet model.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to train for.")

    # Dataset params.
    parser.add_argument("--dataset_dir", type=str, default="data/dogs-vs-cats/", help="Path to dataset.")
    parser.add_argument("--img_size", type=int, default=256, help="Size of images to use.")

    args = parser.parse_args()

    main(args)
