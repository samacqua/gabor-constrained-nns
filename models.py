"""Load the base models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _pair

N_OUT = 8


class GaborConv2d(_ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, device="cpu", stride=1,
                 padding=0, dilation=1, groups=1, bias=False, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(GaborConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False,
                                          _pair(0), groups, bias, padding_mode)
        self.freq = nn.Parameter(
            (3.14 / 2) * 1.41 ** (-torch.randint(0, 5, (out_channels, in_channels))).type(torch.Tensor))
        self.theta = nn.Parameter((3.14 / 8) * torch.randint(0, 8, (out_channels, in_channels)).type(torch.Tensor))
        self.psi = nn.Parameter(3.14 * torch.rand(out_channels, in_channels))
        self.sigma = nn.Parameter(3.14 / self.freq)
        self.x0 = torch.ceil(torch.Tensor([self.kernel_size[0] / 2]))[0]
        self.y0 = torch.ceil(torch.Tensor([self.kernel_size[1] / 2]))[0]
        self.device = device

    def forward(self, input_image):
        weight = self.calc_weight()
        return F.conv2d(input_image, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def calc_weight(self):
        y, x = torch.meshgrid([torch.linspace(-self.x0 + 1, self.x0, self.kernel_size[0]),
                               torch.linspace(-self.y0 + 1, self.y0, self.kernel_size[1])])
        x = x.to(self.device)
        y = y.to(self.device)
        weight = torch.empty(self.weight.shape, requires_grad=False).to(self.device)
        for i in range(self.out_channels):
            for j in range(self.in_channels):
                sigma = self.sigma[i, j].expand_as(y)
                freq = self.freq[i, j].expand_as(y)
                theta = self.theta[i, j].expand_as(y)
                psi = self.psi[i, j].expand_as(y)

                rotx = x * torch.cos(theta) + y * torch.sin(theta)
                roty = -x * torch.sin(theta) + y * torch.cos(theta)

                g = torch.zeros(y.shape)

                g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + 1e-3) ** 2))
                g = g * torch.cos(freq * rotx + psi)
                g = g / (2 * 3.14 * sigma ** 2)
                weight[i, j] = g
                self.weight.data[i, j] = g

        return weight


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
