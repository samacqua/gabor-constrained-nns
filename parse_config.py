"""Loads the experiment configuration file."""

import torch
import numpy as np
import yaml
import os
import sys

import torchvision
import torchvision.transforms as transforms

from src.models import CNN, CNNSmall, CNNLinear


def parse_config(config_path: str):
    """Parses the configuration file."""

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Set the random seed.
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Load the datasets.
    config['datasets']['initial'], config['datasets']['finetune'] = load_datasets(config['datasets'])

    # Make the save directories.
    os.makedirs(config['save_dir'], exist_ok=True)

    default_schedule_params = {
        "gabor_constrained": False, "skip": False, "freeze_first_layer": False, "optimizer_params": {},
    }

    # Set the default sequence parameters.
    for schedule_name, training_schedule in config['schedules'].items():
        stages = {"initial_train", "finetune"}
        assert set(training_schedule.keys()) == stages, \
            "Training schedule must have 'initial_train' and 'finetune' keys."

        for stage in stages:
            training_schedule[stage] = {**default_schedule_params, **training_schedule[stage]}

        training_schedule['model'] = load_model(config['base_model'], 
                                                training_schedule['initial_train']['gabor_constrained'], 
                                                n_channels=config['datasets']['params']['n_channels'])

    return config


def load_datasets(dataset_cfg: dict) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Loads a train + test dataset."""

    img_size = dataset_cfg['params']['img_size']
    n_channels = dataset_cfg['params']['n_channels']

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

    # Load each dataset.
    loaded_datasets = []
    for dataset_name in (dataset_cfg['initial'], dataset_cfg['finetune']):
        if dataset_name not in datasets:
            raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets: {list(datasets.keys())}")
        
        dataset = datasets[dataset_name]
        trainset = dataset['loader'](root='./data', train=True,
                                                download=True, transform=dataset['transform'])
        testset = dataset['loader'](root='./data', train=False,
                                            download=True, transform=dataset['transform'])
        
        loaded_datasets.append((trainset, testset))
    
    return loaded_datasets
    


def load_model(base_model_config: dict, is_gabornet: bool, n_channels: int) -> CNN:
    """Loads the base model."""

    model_arch = getattr(sys.modules[__name__], base_model_config['name'])
    return model_arch(is_gabornet, n_channels=n_channels, kernel_size=base_model_config['kernel_size'])
