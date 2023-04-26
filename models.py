"""Load the base models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from GaborNet import GaborConv2d

N_OUT = 8

class GaborBase(nn.Module):
    """Base class that allows switching between CNN and GaborNet."""

    def __init__(self):
        super().__init__()

    def unconstrain(self):
        """Makes the first layer a GaborNet or standard convolutional layer."""

        # Return if already unconstrained.
        if not self.is_gabornet:
            return
        
        # Get current parameters.
        old_params = self.conv1.state_dict()

        # Create a new layer and use old weights.
        self.conv1 = nn.Conv2d(self.n_channels, N_OUT, self.kernel_size)
        new_params = self.conv1.state_dict()
        new_params['weight'] = old_params['weight']
        new_params['bias'] = torch.zeros_like(new_params['bias'])
        self.conv1.load_state_dict(new_params)

        self.is_gabornet = False

    def get_conv_weights(self) -> torch.Tensor:
        """Gets the weights of the first convolutional layer."""

        # Since the weights aren't updated directly (the parameters of the Gabor equation are), we need to update the
        # weights before returning them.
        if self.is_gabornet:
            return self.conv1.calc_weight()

        return self.conv1.weight
    
    def freeze_first_layer(self):
        """Freezes the first layer."""

        self.conv1.weight.requires_grad = False

        # Need to freeze the Gabor parameters.
        if self.is_gabornet:
            self.conv1.freq.requires_grad = False
            self.conv1.theta.requires_grad = False
            self.conv1.psi.requires_grad = False
            self.conv1.sigma.requires_grad = False

        else:
            self.conv1.bias.requires_grad = False


class CNN(GaborBase):
    """A simple CNN that can have a gabor-constrained first layer."""
        
    def __init__(self, is_gabornet: bool = False, n_channels: int = 3, kernel_size: int = 10):
        super().__init__()
        n_classes = 10
        conv1_type = GaborConv2d if is_gabornet else nn.Conv2d
        self.conv1 = conv1_type(n_channels, N_OUT, kernel_size=(kernel_size, kernel_size), stride=1)
        self.c1 = nn.Conv2d(N_OUT, N_OUT*2, kernel_size=(3, 3), stride=2)
        self.c2 = nn.Conv2d(N_OUT*2, N_OUT*4, kernel_size=(3, 3), stride=2)
        self.fc1 = nn.LazyLinear(128)
        self.fc3 = nn.Linear(128, n_classes)

        self.is_gabornet = is_gabornet
        self.n_channels = n_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        x = F.max_pool2d(F.leaky_relu(self.c1(x)), kernel_size=2)
        x = F.max_pool2d(F.leaky_relu(self.c2(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = nn.Dropout()(x)
        x = self.fc3(x)
        return x
    

class CNNSmall(GaborBase):
    """An even smaller CNN that can have a gabor-constrained first layer."""
        
    def __init__(self, is_gabornet: bool = False, n_channels: int = 3, kernel_size: int = 10):
        super().__init__()
        n_classes = 10
        conv1_type = GaborConv2d if is_gabornet else nn.Conv2d
        self.conv1 = conv1_type(n_channels, N_OUT, kernel_size=(kernel_size, kernel_size), stride=1)
        self.c1 = nn.Conv2d(N_OUT, N_OUT*2, kernel_size=(3, 3), stride=2)
        self.fc1 = nn.LazyLinear(64)
        self.fc3 = nn.Linear(64, n_classes)

        self.is_gabornet = is_gabornet
        self.n_channels = n_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.conv1(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        x = F.max_pool2d(F.leaky_relu(self.c1(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        x = torch.flatten(x, 1)
        x = F.leaky_relu(self.fc1(x))
        x = nn.Dropout()(x)
        x = self.fc3(x)
        return x


class CNNLinear(GaborBase):
    """A linear CNN that can have a gabor-constrained first layer."""

    def __init__(self, is_gabornet: bool = False, n_channels: int = 3, kernel_size: int = 10):
        super().__init__()
        n_classes = 10
        conv1_type = GaborConv2d if is_gabornet else nn.Conv2d
        self.conv1 = conv1_type(n_channels, N_OUT, kernel_size=(kernel_size, kernel_size), stride=1)
        self.fc1 = nn.LazyLinear(n_classes)

        self.is_gabornet = is_gabornet
        self.n_channels = n_channels
        self.kernel_size = kernel_size

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x))
        x = torch.flatten(x, 1)
        x = self.fc1(x)

        return x


if __name__ == "__main__":
    pass
