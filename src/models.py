"""Load the base models."""

from typing import Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parameter import Parameter

from gabor_layer import GaborConv2dPip, GaborConv2dGithub, GaborConv2dGithubUpdate

N_CHANNELS_OUT = 8


def load_net(checkpoint: dict, model: "GaborBase", optimizer: optim.Optimizer = None, strict: bool = True
             ) -> tuple[nn.Module, optim.Optimizer, int]:
    """Loads a model from a file, with the optimizer and epoch."""

    # Load the model.
    if model.is_gabornet and isinstance(model.g1, (GaborConv2dGithub, GaborConv2dGithubUpdate)):
        # Needed to avoid some memory issues.
        model.g1.x = Parameter(model.g1.x.contiguous())
        model.g1.y = Parameter(model.g1.y.contiguous())
        model.g1.x_grid = Parameter(model.g1.x_grid.contiguous())
        model.g1.y_grid = Parameter(model.g1.y_grid.contiguous())

    # Make sure bias is set correctly.
    if isinstance(model.g1, nn.Conv2d) and "g1.bias" in checkpoint["model_state_dict"]:
        model.g1.bias = Parameter(torch.zeros_like(checkpoint["model_state_dict"]["g1.bias"]))

    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load the optimizer.
    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return model, optimizer, checkpoint["epoch"]


class GaborBase(nn.Module):
    """Base class that allows switching between CNN and GaborNet."""

    def __init__(self, is_gabornet: bool = False, gabor_type = GaborConv2dPip, n_channels: int = 3, 
                 kernel_size: int = 5, bias: bool = False, device="cpu"):
        super().__init__()

        self.is_gabornet = is_gabornet

        if is_gabornet:
            self.g1 = gabor_type(n_channels, N_CHANNELS_OUT, kernel_size=(kernel_size, kernel_size), stride=1, device=device)
        else:
            self.g1 = nn.Conv2d(n_channels, N_CHANNELS_OUT, kernel_size=(kernel_size, kernel_size), stride=1, bias=bias)


    def unconstrain(self):
        """Makes the first layer into a standard convolutional layer."""

        # Return if already unconstrained.
        if not self.is_gabornet:
            return
        
        # Get current parameters.
        old_params = self.g1.state_dict()

        # Create a new layer and use old weights.
        self.g1 = nn.Conv2d(self.n_channels, N_CHANNELS_OUT, self.kernel_size)
        new_params = self.g1.state_dict()
        new_params['weight'] = old_params['weight']
        new_params['bias'] = torch.zeros_like(new_params['bias'])
        self.g1.load_state_dict(new_params)

        self.is_gabornet = False

    def get_conv_weights(self) -> torch.Tensor:
        """Gets the weights of the first convolutional layer."""

        # Since the weights aren't updated directly (the parameters of the Gabor equation are), we need to update the
        # weights before returning them.
        if self.is_gabornet:
            return self.g1.calculate_weights()

        return self.g1.weight
    
    def _freeze_layer(self, layer: nn.Module):
        """Freezes the layer."""

        layer.weight.requires_grad = False

        # Need to freeze the Gabor parameters.
        try:
            layer.freq.requires_grad = False
            layer.theta.requires_grad = False
            layer.psi.requires_grad = False
            layer.sigma.requires_grad = False

        except:
            layer.bias.requires_grad = False

    def freeze_first_layer(self):
        """Freezes the first layer."""
        self._freeze_layer(self.g1)


class CNN(GaborBase):
    """A simple CNN that can have a gabor-constrained first layer."""
        
    def __init__(self, is_gabornet: bool = False, n_channels: int = 3, kernel_size: int = 10, device = 'cpu',
                 add_padding: bool = False, gabor_type = GaborConv2dPip, bias=False, n_classes: int = 10):
        super().__init__(is_gabornet=is_gabornet, gabor_type=gabor_type, n_channels=n_channels, kernel_size=kernel_size,
                        bias=bias, device=device)

        self.c1 = nn.Conv2d(N_CHANNELS_OUT, N_CHANNELS_OUT*2, kernel_size=(3, 3), stride=1)
        self.c2 = nn.Conv2d(N_CHANNELS_OUT*2, N_CHANNELS_OUT*4, kernel_size=(3, 3), stride=1)
        self.c3 = nn.Conv2d(N_CHANNELS_OUT*4, N_CHANNELS_OUT*4, kernel_size=(3, 3), stride=1)

        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, n_classes)

        self.dropout2d = nn.Dropout2d()
        self.dropout = nn.Dropout()


    def forward(self, x):

        x = F.max_pool2d(F.leaky_relu(self.g1(x)), kernel_size=2)
        x = self.dropout2d(x)

        x = F.max_pool2d(F.leaky_relu(self.c1(x)), kernel_size=2)
        x = self.dropout2d(x)

        x = F.max_pool2d(F.leaky_relu(self.c2(x)), kernel_size=2)
        x = self.dropout2d(x)

        x = F.max_pool2d(F.leaky_relu(self.c3(x)), kernel_size=2)
        x = self.dropout2d(x)

        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x
    

class CNNSmall(GaborBase):
    """An even smaller CNN that can have a gabor-constrained first layer."""
        
    def __init__(self, is_gabornet: bool = False, n_channels: int = 3, kernel_size: int = 10, device = 'cpu',
                 add_padding: bool = False, gabor_type = GaborConv2dPip, bias=False, n_classes: int = 10):
        super().__init__(is_gabornet=is_gabornet, gabor_type=gabor_type, n_channels=n_channels, kernel_size=kernel_size,
                        bias=bias, device=device)

        self.fc1 = nn.LazyLinear(64)
        self.fc2 = nn.Linear(64, n_classes)

        self.dropout2d = nn.Dropout2d()
        self.dropout = nn.Dropout()

    def forward(self, x):

        x = F.max_pool2d(F.leaky_relu(self.g1(x)), kernel_size=2)
        x = self.dropout2d(x)
        
        x = F.max_pool2d(F.leaky_relu(self.c1(x)), kernel_size=2)
        x = self.dropout2d(x)
        
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x


class CNNLinear(GaborBase):
    """A single linear layer after one convolutional layer that can be gabor-constrained."""

    def __init__(self, is_gabornet: bool = False, n_channels: int = 3, kernel_size: int = 10, device = 'cpu',
                 add_padding: bool = False, gabor_type = GaborConv2dPip, bias=False, n_classes: int = 10):
        super().__init__(is_gabornet=is_gabornet, gabor_type=gabor_type, n_channels=n_channels, kernel_size=kernel_size,
                        bias=bias, device=device)
        
        self.fc1 = nn.LazyLinear(n_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):

        x = F.max_pool2d(F.relu(self.g1(x)), kernel_size=2)
        x = self.dropout(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    pass
