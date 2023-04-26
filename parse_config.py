"""Loads the experiment configuration file."""

import torch
import numpy as np
import yaml
import os
import sys

import torchvision
import torchvision.transforms as transforms

from models import CNN, CNNSmall, CNNLinear


def parse_config(config_path: str):
    """Parses the configuration file."""

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set the random seed.
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Load the base model.
    config['base_model'] = load_model(config['base_model'])

    # Load the datasets.
    config['initial_dataset'] = load_dataset(config['initial_dataset'], config['n_channels'], config['img_size'])
    config['finetune_dataset'] = load_dataset(config['finetune_dataset'], config['n_channels'], config['img_size'])

    # Make the save directory.
    os.makedirs(config['save_dir'], exist_ok=True)

    return config


def load_dataset(dataset_name: str, n_channels: int = 1, img_size: int = 32) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Loads a train + test dataset."""

    # Make the transforms based on the number of channels.
    # CIFAR-10 is 3-channel, Fashion-MNIST is 1-channel, so 1 has to change.
    # If 3 channels, then just copying Fashion-MNIST across all channels.
    # If 1 channel, then converting CIFAR-10 to grayscale.
    cifar_transforms = ([torchvision.transforms.Grayscale(num_output_channels=1)] if n_channels == 1 else []) + [
        torchvision.transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * n_channels, (0.5,) * n_channels)
    ]
    fashion_mnist_transforms = ([torchvision.transforms.Grayscale(num_output_channels=3)] if n_channels == 3 else []) + [
        torchvision.transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * n_channels, (0.5,) * n_channels)
    ]

    # The transforms normalize each to a 3-channel 32x32 image.
    datasets = {
        "cifar10": {
            "loader": torchvision.datasets.CIFAR10, 
            "transform": transforms.Compose(cifar_transforms)
        },
        "fashion_mnist": {
            "loader": torchvision.datasets.FashionMNIST, 
            "transform": transforms.Compose(fashion_mnist_transforms)
        },
    }

    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets: {list(datasets.keys())}")
    
    dataset = datasets[dataset_name]
    trainset = dataset['loader'](root='./data', train=True,
                                            download=True, transform=dataset['transform'])
    testset = dataset['loader'](root='./data', train=False,
                                        download=True, transform=dataset['transform'])
    
    return trainset, testset
    


def load_model(model_config: dict) -> CNN:
    """Loads the base model."""

    # Load the model.
    return lambda *args, **kwargs: getattr(sys.modules[__name__], model_config['name'])(
        *args, kernel_size=model_config['kernel_size'], **kwargs)
