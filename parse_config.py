"""Loads the experiment configuration file."""

import torch
import numpy as np
import yaml
import os
import sys

from src.models import CNN, CNNSmall, CNNLinear
from src.datasets import load_dataset


def parse_config(config_path: str, config_updates: dict = None):
    """Parses the configuration file."""

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Update the configuration.
    if config_updates is not None:
        config = {**config, **config_updates}

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
    dataset_dir = dataset_cfg['params'].get('dataset_dir', './data')

    # Load each dataset.
    loaded_datasets = []
    for dataset_name in (dataset_cfg['initial'], dataset_cfg['finetune']):
        trainset, testset, _ = load_dataset(dataset_name, img_size, n_channels, dataset_dir)        
        loaded_datasets.append((trainset, testset))
    
    return loaded_datasets
    


def load_model(base_model_config: dict, is_gabornet: bool, n_channels: int) -> CNN:
    """Loads the base model."""

    model_arch = getattr(sys.modules[__name__], base_model_config['name'])
    return model_arch(is_gabornet, n_channels=n_channels, kernel_size=base_model_config['kernel_size'])
