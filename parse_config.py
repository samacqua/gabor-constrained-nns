"""Loads the experiment configuration file."""

import torch
import yaml
import os

import torchvision
import torchvision.transforms as transforms

from models import CNN


def parse_config(config_path: str):
    """Parses the configuration file."""

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Load the base model.
    config['base_model'] = load_model(config['base_model'])

    # Load the datasets.
    config['initial_dataset'] = load_dataset(config['initial_dataset'])
    config['finetune_dataset'] = load_dataset(config['finetune_dataset'])

    # Make the save directory.
    os.makedirs(config['save_dir'], exist_ok=True)

    return config


def load_dataset(dataset_name: str) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Loads a train + test dataset."""

    # The transforms normalize each to a 3-channel 32x32 image.
    datasets = {
        "cifar10": {
            "loader": torchvision.datasets.CIFAR10, 
            "transform": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
        },
        "fashion_mnist": {
            "loader": torchvision.datasets.FashionMNIST, 
            "transform": transforms.Compose([
                torchvision.transforms.Grayscale(num_output_channels=3),
                torchvision.transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])
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
    model = CNN()

    return model

