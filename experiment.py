"""Runs experiments."""

import argparse
from copy import deepcopy
import os

import torch

from train import train
from parse_config import parse_config

import warnings
warnings.filterwarnings("ignore")


def run_experiment(config: dict):
    """Runs an experiment."""

    out_dir = config['save_dir']
    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss()
    save_every = config['save_every']

    # Load the base model and dataset.
    base_model = config['base_model']
    dataset_a_train, dataset_a_test = config['initial_dataset']
    dataset_b_train, dataset_b_test = config['finetune_dataset']

    for schedule_name, training_schedule in config['schedules'].items():

        print()
        print(schedule_name)

        model_save_dir = os.path.join(out_dir, "models", schedule_name)
        os.makedirs(model_save_dir, exist_ok=True)

        initial_params = training_schedule['initial_train']

        # Load the initial model and dataloader.
        model = base_model(is_gabornet=initial_params['gabor_constrained'], n_channels=config['n_channels'])
        trainloader_a = torch.utils.data.DataLoader(dataset_a_train, **config['dataloader_params'])
        log_dir = os.path.join(out_dir, "logs", schedule_name + "_a")
        os.makedirs(log_dir, exist_ok=True)

        # Train the model on the first dataset.
        opt = torch.optim.Adam(model.parameters(), **initial_params.get('optimizer_params', {}))
        pre_train_a_weights = deepcopy(model.get_conv_weights().detach().clone())
        train(model, trainloader_a, criterion=criterion, device=device, optimizer=opt, log_dir=log_dir, 
              save_every=save_every, model_save_dir=model_save_dir, model_suffix='a', epochs=config['epochs'])
        
        # Check that the weights changed.
        pre_train_b_weights = deepcopy(model.get_conv_weights().detach().clone())
        assert not torch.allclose(pre_train_a_weights, pre_train_b_weights, atol=1e-6), "Weights didn't change!"

        # Prep the model for finetuning.
        finetune_params = training_schedule['finetune']
        if not finetune_params['gabor_constrained'] and model.is_gabornet:
            model.unconstrain()
        if finetune_params['freeze_first_layer']:
            model.freeze_first_layer()
        trainloader_b = torch.utils.data.DataLoader(dataset_b_train, **config['dataloader_params'])
        log_dir = os.path.join(out_dir, "logs", schedule_name + "_b")
        os.makedirs(log_dir, exist_ok=True)

        # Train the model on the second dataset.
        opt = torch.optim.Adam(model.parameters(), **finetune_params.get('optimizer_params', {}))
        train(model, trainloader_b, criterion=criterion, device=device, optimizer=opt, log_dir=log_dir, 
              save_every=save_every, model_save_dir=model_save_dir, model_suffix='b', epochs=config['epochs'])
        
        # Check that the first layer weights are actually frozen.
        post_train_weights = deepcopy(model.get_conv_weights().detach().clone())
        weights_changed = torch.allclose(pre_train_b_weights, post_train_weights, atol=1e-6)
        assert weights_changed == finetune_params['freeze_first_layer'], \
            f"Expected 1st layer to change? {not finetune_params['freeze_first_layer']}. Changed? {not weights_changed}"


def main():

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    # Parse the configuration file + run the experiment.
    config = parse_config(args.config)

    # If repeating an experiment multiple times, then fix the directories.
    og_save_dir = config['save_dir']
    for i in range(config['n_repeats']):
        config['save_dir'] = os.path.join(og_save_dir, f"repeat_{i+1}")
        print(f"Running repeat {i+1} of {config['n_repeats']}")
        run_experiment(config)


if __name__ == "__main__":
    main()
