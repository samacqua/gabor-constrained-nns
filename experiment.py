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

    # Load the base model and dataset.
    base_model = config['base_model']
    dataset_a_train, dataset_a_test = config['initial_dataset']
    dataset_b_train, dataset_b_test = config['finetune_dataset']

    for schedule_name, training_schedule in config['schedules'].items():

        model_save_dir = os.path.join(out_dir, "models", schedule_name)
        os.makedirs(model_save_dir, exist_ok=True)

        print(training_schedule)
        initial_params = training_schedule['initial_train']

        # Load the initial model and dataloader.
        model = base_model(is_gabornet=initial_params['gabor_constrained'], n_channels=config['n_channels'])
        trainloader_a = torch.utils.data.DataLoader(dataset_a_train, **config['dataloader_params'])
        log_dir = os.path.join(out_dir, "logs", schedule_name + "_a")
        os.makedirs(log_dir, exist_ok=True)

        # Train the model on the first dataset.
        opt = torch.optim.Adam(model.parameters(), **initial_params.get('optimizer_params', {}))
        train(model, trainloader_a, criterion=criterion, device=device, optimizer=opt, log_dir=log_dir)
        model_a_save_path = os.path.join(model_save_dir, 'model_a.pt')
        torch.save(model.state_dict(), model_a_save_path)

        # Prep the model for finetuning.
        finetune_params = training_schedule['finetune']
        if not finetune_params['gabor_constrained'] and model.is_gabornet:
            model.unconstrain()
        model.conv1.weight.requires_grad = not finetune_params['freeze_first_layer']
        trainloader_b = torch.utils.data.DataLoader(dataset_b_train, **config['dataloader_params'])
        log_dir = os.path.join(out_dir, "logs", schedule_name + "_b")
        os.makedirs(log_dir, exist_ok=True)

        # Train the model on the second dataset.
        opt = torch.optim.Adam(model.parameters(), **finetune_params.get('optimizer_params', {}))
        pre_train_weights = deepcopy(model.get_conv_weights())
        train(model, trainloader_b, criterion=criterion, device=device, optimizer=opt, log_dir=log_dir)

        # Check that the first layer weights are actually frozen.
        post_train_weights = deepcopy(model.get_conv_weights())
        if finetune_params['freeze_first_layer']:
            assert torch.allclose(pre_train_weights, post_train_weights, atol=1e-6)

        model_b_save_path = os.path.join(model_save_dir, 'model_b.pt')
        torch.save(model.state_dict(), model_b_save_path)


def main():

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    # Parse the configuration file + run the experiment.
    config = parse_config(args.config)
    run_experiment(config)


if __name__ == "__main__":
    main()
