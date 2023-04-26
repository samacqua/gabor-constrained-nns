"""Load the base models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from GaborNet import GaborConv2d


class CNN(nn.Module):
    """A simple CNN that can have a gabor-constrained first layer."""
        
    def __init__(self, is_gabornet: bool = False, n_channels: int = 3):
        super().__init__()
        filter_size = 5
        self.conv1 = GaborConv2d(n_channels, 6, filter_size) if is_gabornet else nn.Conv2d(n_channels, 6, filter_size)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.is_gabornet = is_gabornet
        self.n_channels = n_channels

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def unconstrain(self):
        """Makes the first layer a GaborNet or standard convolutional layer."""

        # Return if already unconstrained.
        if not self.is_gabornet:
            return
        
        # Get current parameters.
        old_params = self.conv1.state_dict()
        from copy import deepcopy
        old_conv = deepcopy(self.conv1)

        # Create a new layer and use old weights.
        self.conv1 = nn.Conv2d(self.n_channels, 6, 5)
        new_params = self.conv1.state_dict()
        new_params['weight'] = old_params['weight']
        new_params['bias'] = torch.zeros_like(new_params['bias'])
        self.conv1.load_state_dict(new_params)

        self.is_gabornet = False


if __name__ == "__main__":
    pass