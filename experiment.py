"""Runs experiments."""

import argparse
from copy import deepcopy
import os

import torch

from train import train
from parse_config import parse_config

import warnings
warnings.filterwarnings("ignore")


def run_experiment_leg(base_model, config):
    """Runs part of the experiment."""


def run_experiment(config: dict):
    """Runs an experiment."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = torch.nn.CrossEntropyLoss()

    # Load training parameters.
    out_dir = config['save_dir']
    training_cfg = config['training']
    save_every = training_cfg['save_every']
    n_epochs = training_cfg['epochs']

    # Load the dataset.
    dataset_cfg = config['datasets']
    dataset_a_train, dataset_a_test = dataset_cfg['initial']
    dataset_b_train, dataset_b_test = dataset_cfg['finetune']
    n_channels = dataset_cfg['params']['n_channels']

    # Load the base model and dataset.
    base_model = config['base_model']

    for schedule_name, training_schedule in config['schedules'].items():

        print("\n" + "=== " + schedule_name + " ===")

        model_save_dir = os.path.join(out_dir, "models", schedule_name)
        os.makedirs(model_save_dir, exist_ok=True)

        initial_params = training_schedule['initial_train']
        model = base_model(is_gabornet=initial_params['gabor_constrained'], n_channels=n_channels, device=device)
        
        # Train the model on the first dataset, then on the second dataset.
        for i, (exp_params, dataset) in enumerate((
            (training_schedule['initial_train'], dataset_a_train),
            (training_schedule['finetune'], dataset_b_train))):

            # Set the model parameters.
            if not exp_params['gabor_constrained'] and model.is_gabornet:
                model.unconstrain()
            if exp_params['freeze_first_layer']:
                model.freeze_first_layer()

            # Load the dataset.
            trainloader = torch.utils.data.DataLoader(dataset, **training_cfg['dataloader_params'])
            dataset_name = "a" if i == 0 else "b"
            log_dir = os.path.join(out_dir, "logs", schedule_name + "_" + dataset_name)
            os.makedirs(log_dir, exist_ok=True)

            # Train the model on the first dataset.
            opt = torch.optim.Adam(model.parameters(), **exp_params['optimizer_params'])
            pre_train_weights = deepcopy(model.get_conv_weights().detach().clone())
            if not exp_params['skip']:
                print("Training on dataset " + dataset_name.upper())
                train(model, trainloader, criterion=criterion, device=device, optimizer=opt, log_dir=log_dir, 
                    save_every=save_every, model_save_dir=model_save_dir, model_suffix=dataset_name, 
                    epochs=n_epochs)
        
            # Check that the weights changed.
            post_train_weights = deepcopy(model.get_conv_weights().detach().clone())
            weights_changed = not torch.allclose(pre_train_weights, post_train_weights, atol=1e-6)
            weights_should_change = (not exp_params['skip'] and not exp_params['freeze_first_layer'])
            assert weights_changed == weights_should_change, \
                f"Weights should{'' if weights_should_change else ' not'} have changed. Changed? {weights_changed}"


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
