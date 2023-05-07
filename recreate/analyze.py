"""Analyze the results of the recreation experiment."""

import argparse
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
from typing import Type
import json
import sys

import torch
from torch.utils.data import DataLoader

from .dogs_cats import DogCatNNSanity, DogCatNet, load_dataset, load_net
from analysis import test_accuracy
from gabor_layer import GaborConv2d, GaborConv2dBuggy, GaborConv2dStillBuggy


def make_accuracy_fig(gabor_train: dict[int, float], cnn_train: dict[int, float], gabor_test: dict[int, float], 
             cnn_test: dict[int, float], gabor_train_reported: dict[int, float] = None, 
             cnn_train_reported: dict[int, float] = None, gabor_test_reported: dict[int, float] = None,
             cnn_test_reported: dict[int, float] = None, save_dir: str = None) -> None:
    """Makes a figure of accuracy in the same way as the original paper."""

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 7))

    # Plot the train accuracies of the model we trained / evaluated.
    gabor_train_x, gabor_train_y = zip(*sorted(gabor_train.items()))
    cnn_x, cnn_y = zip(*sorted(cnn_train.items()))

    ax0.plot(gabor_train_x, gabor_train_y, label='Gabor CNN', c='orange')
    ax0.plot(cnn_x, cnn_y, label='CNN', c='blue')

    # Plot the train accuracies of the model reported in the paper.
    if gabor_train_reported and cnn_train_reported:
        gabor_x, gabor_y = zip(*sorted(gabor_train_reported.items()))
        cnn_x, cnn_y = zip(*sorted(cnn_train_reported.items()))
        ax0.plot(gabor_x, gabor_y, label='Gabor CNN (reported)', c='orange', linestyle='dashed')
        ax0.plot(cnn_x, cnn_y, label='CNN (reported)', c='blue', linestyle='dashed')

    ax0.set_title('Train on Dogs vs Cats')
    ax0.set_xlabel('epoch')
    ax0.set_ylabel('accuracy')
    ax0.legend()

    # Plot the test accuracies of the model we trained / evaluated.
    gabor_test_x, gabor_test_y = zip(*sorted(gabor_test.items()))
    cnn_test_x, cnn_test_y = zip(*sorted(cnn_test.items()))

    ax1.plot(gabor_test_x, gabor_test_y, label='Gabor CNN', c='orange')
    ax1.plot(cnn_test_x, cnn_test_y, label='CNN', c='blue')

    # Plot the test accuracies of the model reported in the paper.
    if gabor_test_reported and cnn_test_reported:
        gabor_x, gabor_y = zip(*sorted(gabor_test_reported.items()))
        cnn_x, cnn_y = zip(*sorted(cnn_test_reported.items()))
        ax1.plot(gabor_x, gabor_y, label='Gabor CNN (reported)', c='orange', linestyle='dashed')
        ax1.plot(cnn_x, cnn_y, label='CNN (reported)', c='blue', linestyle='dashed')

    ax1.set_title('Test on Dogs vs Cats')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "accuracy.png"))

    plt.show()


def calc_accuracies(models: dict[int, torch.nn.Module], dataloader: DataLoader, cache_path: str = None, 
                    device: str = "cpu", pbar=None, use_cache: bool = False) -> dict[int, float]:
    """Calculates the accuracy of a single model over the course of training on 1 dataloader."""

    accuracies = {}
    pbar = tqdm(total=len(models) * len(dataloader)) if pbar is None else pbar

    # Load from cache if it exists and was calculated with at least as many samples as the current request.
    if cache_path and os.path.exists(cache_path) and use_cache:
        assert cache_path.endswith(".json"), "Cache path must be a JSON file."
        with open(cache_path, "r") as f:
            accuracy_dict = json.load(f)
        if accuracy_dict["n_samples"] >= len(dataloader.dataset):
            accuracies = {int(k): v for k, v in accuracy_dict["accuracies"].items() if int(k) in models}

    for epoch, model in models.items():

        # If we already have the accuracy, skip.
        if epoch in accuracies:
            pbar.update(len(dataloader))
            continue

        # Otherwise, calculate the accuracy.
        accuracies[epoch] = test_accuracy(dataloader, model, device=device, pbar=pbar)

    # Save to cache.
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"n_samples": len(dataloader.dataset), "accuracies": accuracies}, f)

    return accuracies


def calc_accuracies_full(models: dict[str, dict[int, torch.nn.Module]],
               train_loader: DataLoader, test_loader: DataLoader, 
               use_cache: bool = True, device: str = 'cpu'
               ) -> dict[str, tuple[dict[int, float], dict[int, float]]]:
    """Calculates the accuracies over epochs on the train / test datasets for the multiple models.
    
    Args:
        models: {model_name: {"checkpoints":{epoch: model}, "save_dir": save_dir}}
        train_loader: The dataloader for the training dataset.
        test_loader: The dataloader for the test dataset.
        save_dir: The directory to save the accuracies to.
        use_cache: Whether to use the cache or not.
        device: The device to run the models on.

    Returns:
        {model_name: ({epoch: train_accuracy}, {epoch: test_accuracy}})
    """

    # Check input is well formed.
    try:
        _ = [model_info['checkpoints'] for model_info in models.values()]
        _ = [model_info['save_dir'] for model_info in models.values()]
    except KeyError:
        raise ValueError("Models must be a dictionary of {model_name: {'checkpoints': {epoch: model}, 'save_dir': save_dir}}")

    # Determine the size of the progress bar.
    n_checkpoints = sum([len(model_info['checkpoints']) for model_info in models.values()])
    n_iters = n_checkpoints * (len(train_loader) + len(test_loader))

    model_accuracies = {}
    with tqdm(total=n_iters, maxinterval=5) as pbar:
            
        # Calculate the accuracies for each model.
        for model_name, model_info in models.items():

            # Load the checkpoints.
            checkpoints = model_info['checkpoints']
            save_dir = model_info['save_dir']

            # Calculate the accuracies.
            train_accuracies = calc_accuracies(checkpoints, train_loader, cache_path=os.path.join(save_dir, "train_accuracies.json"), 
                                            device=device, pbar=pbar, use_cache=use_cache)
            test_accuracies = calc_accuracies(checkpoints, test_loader, cache_path=os.path.join(save_dir, "test_accuracies.json"), 
                                            device=device, pbar=pbar, use_cache=use_cache)

            # Save the accuracies.
            model_accuracies[model_name] = (train_accuracies, test_accuracies)

    return model_accuracies


def calc_accuracies_cnn_gabor(gabor_models: dict[int, torch.nn.Module], cnn_models: dict[int, torch.nn.Module], 
                              train_loader: DataLoader, test_loader: DataLoader, save_dir: str, use_cache: bool = False, 
                              device: torch.device = 'cpu'):
    """Calculates the accuracy for a gabornet v. cnn model on the train and test datasets."""

    # Format the models: {model_name: {"checkpoints":{epoch: model}, "save_dir": save_dir}}
    models = {
        "gabornet": {"checkpoints": gabor_models, "save_dir": os.path.join(save_dir, "gabornet")},
        "cnn": {"checkpoints": cnn_models, "save_dir": os.path.join(save_dir, "cnn")},
    }

    accuracies = calc_accuracies_full(models, train_loader, test_loader, use_cache=use_cache, device=device)

    # Parse the output: {model_name: {epoch: (train_accuracy, test_accuracy)}}
    gabor_train, gabor_test = accuracies["gabornet"]
    cnn_train, cnn_test = accuracies["cnn"]

    return gabor_train, cnn_train, gabor_test, cnn_test



def update_checkpoint(checkpoint: dict, gabor_type: str | None, kernel_size: int, add_padding: bool, save_path: str
                      ) -> dict:
    """Updates the checkpoint to include the gabor type, kernel size, and padding. Used to update old checkpoints."""

    checkpoint["gabor_type"] = gabor_type
    checkpoint["kernel_size"] = kernel_size
    checkpoint["add_padding"] = add_padding

    # Save the checkpoint.
    torch.save(checkpoint, save_path)

    return checkpoint


def load_models(base_model: Type[torch.nn.Module], epochs_to_load: set[int] | None, models_dir: str, 
                device: torch.device, calc_weights: bool, train_args: dict = None):
    """Loads the model checkpoints from the directory.
    
    Args:
        base_model: The base model class.
        epochs_to_load: The epochs to load. If None, loads all epochs.
        models_dir: The directory to load the models from.
        device: The device to load the models on.
        calc_weights: Whether to calculate the weights for the model.
        train_args: The arguments passed to the training script. Used to update the model checkpoint if it does not
            contain the necessary information. TODO: deprecate once all are updated.

    Returns:
        {epoch: model}
    """

    models = {}
    for fname in os.listdir(models_dir):
        epoch = int(fname.split("epoch_")[-1].split(".")[0])    # Files of format "epoch_{epoch}.pth"

        # Only load those from the specified epochs.
        if epochs_to_load is not None and epoch not in epochs_to_load:
            continue

        checkpoint_path = os.path.join(models_dir, fname)
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Update the checkpoint if necessary.
        if "gabor_type" not in checkpoint or "kernel_size" not in checkpoint or "add_padding" not in checkpoint:
            assert train_args['no_padding']

            is_gabornet = "gabornet" in models_dir
            kernel_size = train_args['gabor_kernel'] if is_gabornet else train_args['cnn_kernel']
            gabor_type = train_args['gabor_type'] if is_gabornet else None
            add_padding = False
            
            c = input(f"updating checkpoint {checkpoint_path} with gabor type {gabor_type}, kernel size {kernel_size}, "
                      f" and add padding {add_padding}. Continue? (y/n)")
            if c == "y":
                checkpoint = update_checkpoint(checkpoint, gabor_type, kernel_size, add_padding, checkpoint_path)
            else:
                sys.exit(0)

        # Load the model.
        gabor_type_str = checkpoint["gabor_type"]
        kernel_size = checkpoint["kernel_size"]
        add_padding = checkpoint["add_padding"]

        is_gabornet = gabor_type_str is not None
        gabor_type = globals()[gabor_type_str] if is_gabornet else None

        model = base_model(is_gabornet=is_gabornet, kernel_size=kernel_size, add_padding=add_padding, 
                        gabor_type=gabor_type, device=device)
        model, _, model_epoch = load_net(checkpoint, model, strict=calc_weights)

        # TODO: re-run buggy experiments so this isn't necessary.
        model.calc_weights = calc_weights

        assert epoch == model_epoch
        models[epoch + 1] = model   # Epochs are 0-indexed in files, but we want 1-indexed.

    return models


def load_cnn_gabor_models(base_model: torch.nn.Module = Type[torch.nn.Module], epochs_to_load: set[int] = None,
                gabor_models_dir: str = None, cnn_models_dir: str = None, device: str = 'cpu',
                calc_weights: bool = True, train_args=None,
                ) -> tuple[dict[int, torch.nn.Module], dict[int, torch.nn.Module]]:
    """Loads the Gabor + CNN models."""

    gabor_models = load_models(base_model, epochs_to_load, os.path.join(gabor_models_dir, "gabornet"), device, 
                               calc_weights, train_args)
    cnn_models = load_models(base_model, epochs_to_load, os.path.join(cnn_models_dir, "cnn"), device, calc_weights, 
                             train_args)
    
    assert len(gabor_models) == len(cnn_models)

    return gabor_models, cnn_models


def main():

    # Parse the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_name", type=str, default=None, 
                        help="Name of directory to load models from.")
    parser.add_argument("--seed", type=int, help="The random seed of the experiment to analyze.")

    parser.add_argument("--use_cache", action="store_true", help="If True, then if accuracies are already calculated, "
                                                                    "they will be loaded from cache.")
    parser.add_argument("--N", type=int, default=None, help="The number of samples to use from the train / test sets.")
    parser.add_argument("--calc_every", type=int, default=3, help="The number of epochs to skip between calculating "
                                                                     "the accuracies.")
    parser.add_argument("--epochs", type=int, nargs="+", default=None, help="1-index epochs to calc accuracies for. If "
                                                                            "None, then will default to calc_every.")

    parser.add_argument("--model", type=str, default="DogCatNet", help="The base model to use for the CNNs.",
                        choices=["DogCatNet", "DogCatNNSanity"])
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to dataset.")
    
    parser.add_argument("--gabor_dir", type=str, default=None, 
                        help="Name of directory to load Gabor models from. Defaults to dir_name.")
    parser.add_argument("--cnn_dir", type=str, default=None, 
                        help="Name of directory to load Gabor models from. Defaults to dir_name.")
    
    # TODO: re-run all buggy experiments so this isn't necessary.
    # If we are not re-calculating the weights, we don't care if extra parameters are loaded.
    # This is only necessary because some checkpoints had cuda bug where not everything was saved properly.
    parser.add_argument("--no_weight_calc", action="store_true", help="Flag to not re-calculate weights on forward pass.")

    parser.add_argument("--assert_gabor_frozen", action="store_true", help="If True, then will assert that the weights of the model did not change during training.")
    parser.add_argument("--assert_cnn_frozen", action="store_true", help="If True, then will assert that the weights of the model did not change during training.")
    
    args = parser.parse_args()

    # Load the training arguments.
    save_dir = f"recreate/out/{args.dir_name}/"
    with open(os.path.join(save_dir, "args.json"), "rb") as f:
        train_args = json.load(f)

    gabor_dir = f"recreate/out/{args.gabor_dir or args.dir_name}/"
    cnn_dir = f"recreate/out/{args.cnn_dir or args.dir_name}/"

    base_model = globals()[args.model]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load the testset.
    print("Loading dataset...")
    torch.manual_seed(args.seed)
    train_set, test_set = load_dataset(args.dataset_dir or train_args['dataset_dir'], 
                                       img_size=(train_args['img_size'], train_args['img_size']))

    # Split the trainset into train and test.
    train_set, test_set1 = torch.utils.data.random_split(
        train_set, [int(len(train_set) * 0.7), int(len(train_set) * 0.3)]
    )

    # Limit the dataset size to make the computations faster.
    N_train = args.N if args.N else len(train_set)
    N_val = args.N if args.N else len(test_set1)
    N_test = args.N if args.N else len(test_set)
    train_set, _ = torch.utils.data.random_split(train_set, [N_train, len(train_set) - N_train])
    val_set, _ = torch.utils.data.random_split(test_set1, [N_val, len(test_set1) - N_val])
    test_set, _ = torch.utils.data.random_split(test_set, [N_test, len(test_set) - N_test])
    test_set = val_set      # Pretty sure the original papers "test set" is actually a validation set.

    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)

    # Load the models.
    print("Loading models...")
    epochs = range(0, 100, args.calc_every) if args.epochs is None else args.epochs
    if args.epochs:
        epochs = [epoch - 1 for epoch in epochs]

    gabor_models_dir = os.path.join(gabor_dir, "models")
    cnn_models_dir = os.path.join(cnn_dir, "models")
    gabor_models, cnn_models = load_cnn_gabor_models(base_model, epochs_to_load=epochs, 
                                           gabor_models_dir=gabor_models_dir, cnn_models_dir=cnn_models_dir, 
                                           device=device, calc_weights=not args.no_weight_calc,
                                           train_args=train_args)

    # Show the accuracies.
    if args.assert_gabor_frozen or args.assert_cnn_frozen:
        first_epoch = min(gabor_models.keys())
        for epoch in gabor_models:
            if epoch == first_epoch:
                continue
            
            if args.assert_gabor_frozen:
                for param in ['freq', 'theta', 'psi', 'sigma', 'weight', 'x0', 'y0']:
                    assert torch.allclose(getattr(gabor_models[first_epoch].g1, param), 
                                          getattr(gabor_models[epoch].g1, param))

            if args.assert_cnn_frozen:
                for param in ['weight', 'bias']:
                    assert torch.allclose(getattr(cnn_models[first_epoch].g1, param), 
                                          getattr(cnn_models[epoch].g1, param))

    # Data from the paper.
    gabor_train_reported = {1: 0.503, 3: 0.597, 10: 0.682, 40: 0.747} #, 90: 0.796}
    cnn_train_reported = {1: 0.506, 3: 0.520, 10: 0.613, 40: 0.674} #, 90: 0.732}
    gabor_test_reported = {1: 0.517, 3: 0.620, 10: 0.679, 40: 0.739} #, 90: 0.792}
    cnn_test_reported = {1: 0.503, 3: 0.515, 10: 0.616, 40: 0.668} #, 90: 0.726}

    # Recreate the figure from the original paper with our replication results.
    print(f"Calculating accuracies (based on N={args.N} samples)...")
    accuracy_dir = os.path.join(save_dir, "accuracy")
    gabor_train, cnn_train, gabor_test, cnn_test = calc_accuracies_cnn_gabor(
        gabor_models, cnn_models, train_loader, test_loader, accuracy_dir, use_cache=args.use_cache, device=device)

    print("Making figure...")
    make_accuracy_fig(gabor_train, cnn_train, gabor_test, cnn_test, 
             gabor_train_reported, cnn_train_reported, gabor_test_reported, cnn_test_reported, save_dir)


if __name__ == '__main__':
    main()
