"""Runs experiments."""

import argparse
from copy import deepcopy
import os

import torch

from train import train, train_many
from parse_config import parse_config

import warnings
warnings.filterwarnings("ignore")


def get_checkpoints(save_dir: str, model_names: list[str], models: list[torch.nn.Module], 
                    optimizers: list[torch.optim.Optimizer]) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    """Returns the latest models from a directory."""

    # Load the models.
    max_epoch = None
    for model, name, optimizer in zip(models, model_names, optimizers):

        # Get the latest epoch.
        model_save_dir = os.path.join(save_dir, "models", name)
        if not os.path.exists(model_save_dir):
            assert max_epoch is None
            return models, optimizers, None
        model_files = os.listdir(model_save_dir)
        model_max_epoch = max([int(fname.split("_")[1].split(".pth")[0]) for fname in model_files])
        if max_epoch is None:
            max_epoch = model_max_epoch
        assert max_epoch == model_max_epoch

        # Load the model.
        save_path = os.path.join(model_save_dir, f"epoch_{max_epoch}.pth")
        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return models, optimizers, max_epoch


def run_experiment(config: dict):
    """Runs an experiment."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load training parameters.
    out_dir = config['save_dir']
    training_cfg = config['training']

    # Load the dataset.
    dataset_cfg = config['datasets']
    dataset_a_train, _ = dataset_cfg['initial']
    dataset_b_train, _ = dataset_cfg['finetune']
    # dataset_a_train = torch.utils.data.Subset(dataset_a_train, range(50))
    # dataset_b_train = torch.utils.data.Subset(dataset_b_train, range(50))

    # Load the models.
    model_names = list(config['schedules'])
    models = [config['schedules'][name]['model'] for name in model_names]
    model_infos = [{} for _ in model_names]
    pre_train_weights = [deepcopy(model.get_conv_weights().detach().clone().to(device)) for model in models]

    # Unconstrain + freeze the models if necessary for finetuning.
    for name, model in zip(model_names, models):
        init_params = config['schedules'][name]['initial_train']
        if init_params['freeze_first_layer']:
            model.freeze_first_layer()
        if not init_params['gabor_constrained']:
            model.unconstrain()
        
        # Set the device since we don't at init. (TODO: fix this).
        model.to(device)
        model.g1.device = device

    # Set up the models + data for initial training.
    optimizers_a = [torch.optim.Adam(
        model.parameters(), **config['schedules'][name]['initial_train']['optimizer_params']) 
        for name, model in zip(model_names, models)]
    dataloader_a = torch.utils.data.DataLoader(dataset_a_train, **training_cfg['dataloader_params'])
    save_dir_a = os.path.join(out_dir, 'dataset_a')

    # Resume training if any training has already been done.
    models, optimizers_a, last_epoch = get_checkpoints(save_dir_a, model_names, models, optimizers_a)
    starting_epoch_a = last_epoch + 1 if last_epoch is not None else 0

    # Run the initial training.
    if starting_epoch_a == training_cfg['initial']['epochs']:
        print("\nFinished training on dataset A.")
    else:
        print("\nTraining on dataset A...")
        train_many(models=models, optimizers=optimizers_a, model_names=model_names, 
                model_infos=model_infos, dataloader=dataloader_a, save_dir=save_dir_a, device=device, 
                starting_epoch = starting_epoch_a, n_epochs = training_cfg['initial']['epochs'])

    # Check that weights changed as expected.
    post_train_weights = [deepcopy(model.get_conv_weights().detach().clone()) for model in models]
    for post_train_weight, pre_train_weight, name in zip(post_train_weights, pre_train_weights, model_names):
        if config['schedules'][name]['initial_train']['freeze_first_layer']:
            assert torch.allclose(post_train_weight, pre_train_weight), \
                "Weights should not have changed after initial training."
        else:
            assert not torch.allclose(post_train_weight, pre_train_weight), \
                "Weights should have changed after initial training."
    
    # Unconstrain + freeze the models if necessary for finetuning.
    for name, model in zip(model_names, models):
        finetune_params = config['schedules'][name]['finetune']
        if finetune_params['freeze_first_layer']:
            model.freeze_first_layer()
        if not finetune_params['gabor_constrained']:
            model.unconstrain()

    # Set up the models + data for fine-tuning.
    dataloader_b = torch.utils.data.DataLoader(dataset_b_train, **training_cfg['dataloader_params'])
    optimizers_b = [torch.optim.Adam(
        model.parameters(), **config['schedules'][name]['finetune']['optimizer_params']) 
        for name, model in zip(model_names, models)]
    save_dir_b = os.path.join(out_dir, 'dataset_b')

    # Resume training if any training has already been done.
    models, optimizers_b, last_epoch = get_checkpoints(save_dir_b, model_names, models, optimizers_b)
    starting_epoch_b = last_epoch + 1 if last_epoch is not None else 0
    if starting_epoch_b > 0:
        assert starting_epoch_a == training_cfg['initial']['epochs'], \
            "Cannot resume training on dataset B if training on dataset A was not completed."

    # Run the fine-tuning.
    if starting_epoch_b == training_cfg['finetune']['epochs']:
        print("\nFinished fine-tuning on dataset B.")
    else:
        print("\nFine-tuning on dataset B.")
        train_many(models=models, optimizers=optimizers_b, model_names=model_names, 
                model_infos=model_infos, dataloader=dataloader_b, save_dir=save_dir_b, device=device, 
                starting_epoch = starting_epoch_b, n_epochs = training_cfg['finetune']['epochs'])

    # Check that weights changed as expected.
    post_finetune_weights = [deepcopy(model.get_conv_weights().detach().clone()) for model in models]
    for post_finetune_weight, post_train_weight, name in zip(post_finetune_weights, post_train_weights, model_names):
        if config['schedules'][name]['finetune']['freeze_first_layer']:
            assert torch.allclose(post_finetune_weight, post_train_weight), \
                "Weights should not have changed after fine-tuning."
        else:
            assert not torch.allclose(post_finetune_weight, post_train_weight), \
                "Weights should have changed after fine-tuning."


def main():

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing results instead of continuing.")

    args = parser.parse_args()

    # Parse the configuration file + run the experiment.
    config = parse_config(args.config)
    if args.overwrite:
        os.system(f"rm -rf {config['save_dir']}")

    # Change the seed each repeat so that we get different random initializations + adjust the save directory.
    og_save_dir = config['save_dir']
    og_seed = config['seed']
    for i in range(config['n_repeats']):
        repeat_seed = og_seed + i
        repeat_save_dir = os.path.join(og_save_dir, str(i))

        config = parse_config(args.config, {'seed': repeat_seed, 'save_dir': repeat_save_dir})

        print(f"Running repeat {i+1} of {config['n_repeats']}")
        run_experiment(config)


if __name__ == "__main__":
    main()
