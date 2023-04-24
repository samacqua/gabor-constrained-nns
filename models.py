"""Load the base models."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from GaborNet import GaborConv2d


class CNN(nn.Module):
    """A simple CNN that can have a gabor-constrained first layer."""
        
    def __init__(self, is_gabornet: bool = False):
        super().__init__()
        self.conv1 = GaborConv2d(3, 6, 5) if is_gabornet else nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.is_gabornet = is_gabornet

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def change_constraint(self, gabor_constrained: bool):
        """Makes the first layer a GaborNet or standard convolutional layer."""

        # Return if nothing needs to be changed.
        if (gabor_constrained and self.is_gabornet) or (not gabor_constrained and not self.is_gabornet):
            return
        
        # Get current parameters.
        old_params = self.conv1.state_dict()

        # TODO: handle bias.
        # Create a new layer and use old weights.
        new_layer_type = GaborConv2d if gabor_constrained else nn.Conv2d
        self.conv1 = new_layer_type(3, 6, 5)
        new_params = self.conv1.state_dict()
        new_params['weight'] = old_params['weight']
        self.conv1.load_state_dict(new_params)

        self.is_gabornet = gabor_constrained


if __name__ == "__main__":
    pass