"""Analyze the results of the recreation experiment."""

import argparse
import matplotlib.pyplot as plt
import pickle
import os
from tqdm import tqdm
from typing import Type
import json

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
    gabor_train_x, gabor_train_y = zip(*gabor_train.items())
    cnn_x, cnn_y = zip(*cnn_train.items())

    ax0.plot(gabor_train_x, gabor_train_y, label='Gabor CNN', c='orange')
    ax0.plot(cnn_x, cnn_y, label='CNN', c='blue')

    # Plot the train accuracies of the model reported in the paper.
    if gabor_train_reported and cnn_train_reported:
        gabor_x, gabor_y = zip(*gabor_train_reported.items())
        cnn_x, cnn_y = zip(*cnn_train_reported.items())
        ax0.plot(gabor_x, gabor_y, label='Gabor CNN (reported)', c='orange', linestyle='dashed')
        ax0.plot(cnn_x, cnn_y, label='CNN (reported)', c='blue', linestyle='dashed')

    ax0.set_title('Train on Dogs vs Cats')
    ax0.set_xlabel('epoch')
    ax0.set_ylabel('accuracy')
    ax0.legend()

    # Plot the test accuracies of the model we trained / evaluated.
    gabor_test_x, gabor_test_y = zip(*gabor_test.items())
    cnn_test_x, cnn_test_y = zip(*cnn_test.items())

    ax1.plot(gabor_test_x, gabor_test_y, label='Gabor CNN', c='orange')
    ax1.plot(cnn_test_x, cnn_test_y, label='CNN', c='blue')

    # Plot the test accuracies of the model reported in the paper.
    if gabor_test_reported and cnn_test_reported:
        gabor_x, gabor_y = zip(*gabor_test_reported.items())
        cnn_x, cnn_y = zip(*cnn_test_reported.items())
        ax1.plot(gabor_x, gabor_y, label='Gabor CNN (reported)', c='orange', linestyle='dashed')
        ax1.plot(cnn_x, cnn_y, label='CNN (reported)', c='blue', linestyle='dashed')

    ax1.set_title('Test on Dogs vs Cats')
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('accuracy')
    ax1.legend()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "accuracy.png"))

    plt.show()


def calc_accuracies(gabor_models: dict[int, torch.nn.Module], cnn_models: dict[int, torch.nn.Module], 
               train_loader: DataLoader, test_loader: DataLoader, save_dir: str, 
               use_cache: bool = True, device: str = 'cpu'
               ) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
    """Calculates the accuracies over epochs on the train / test datasets for the GaborNet and CNN."""

    save_path = os.path.join(save_dir, "accs.pkl")
    epochs = list(sorted(gabor_models.keys()))

    # Load from cache if possible.
    if os.path.exists(save_path) and use_cache:
        print("Loading from cache...")
        with open(save_path, "rb") as f:
            gabor_train_accs, cnn_train_accs, gabor_test_accs, cnn_test_accs = pickle.load(f)

    else:

        # Calculate the accuracies over the train set.
        gabor_train_accs = {}
        cnn_train_accs = {}
        gabor_test_accs = {}
        cnn_test_accs = {}

        n_iters = len(epochs) * (len(train_loader) * 2 + len(test_loader) * 2)
        with tqdm(total=n_iters, maxinterval=5) as pbar:
            for epoch in epochs:
                gabor_train_accs[epoch] = test_accuracy(train_loader, gabor_models[epoch], device=device, pbar=pbar)
                cnn_train_accs[epoch] = test_accuracy(train_loader, cnn_models[epoch], device=device, pbar=pbar)
                gabor_test_accs[epoch] = test_accuracy(test_loader, gabor_models[epoch], device=device, pbar=pbar)
                cnn_test_accs[epoch] = test_accuracy(test_loader, cnn_models[epoch], device=device, pbar=pbar)

        # Save the accuracies.
        with open(save_path, "wb") as f:
            pickle.dump((gabor_train_accs, cnn_train_accs, gabor_test_accs, cnn_test_accs), f)

    return gabor_train_accs, cnn_train_accs, gabor_test_accs, cnn_test_accs


def load_models(base_model: torch.nn.Module = Type[torch.nn.Module], epochs_to_load: set[int] = None,
                gabor_models_dir: str = None, cnn_models_dir: str = None, gabor_type = GaborConv2d, device: str = 'cpu'
                ) -> tuple[dict[int, torch.nn.Module], dict[int, torch.nn.Module]]:
    """Loads the Gabor + CNN models."""

    gabor_models = {}
    cnn_models = {}

    # Load the models by epoch.
    for model_dict, models_dir, is_gabornet in zip((gabor_models, cnn_models), (gabor_models_dir, cnn_models_dir), (True, False)):
        model_name = "gabornet" if is_gabornet else "cnn"

        model_dir = os.path.join(models_dir, model_name)

        for fname in os.listdir(model_dir):
            epoch = int(fname.split("epoch_")[-1].split(".")[0])    # Files of format "epoch_{epoch}.pth"

            # Only load those from the specified epochs.
            if epochs_to_load is not None and epoch not in epochs_to_load:
                continue

            checkpoint = torch.load(os.path.join(model_dir, fname), map_location=device)

            kernel_size = checkpoint.get("kernel_size", (15, 15) if is_gabornet else (5, 5))
            add_padding = checkpoint.get("add_padding", False)
            gabor_type_str = checkpoint.get("gabor_type", None)
            gabor_type = globals()[gabor_type_str] if gabor_type_str else gabor_type

            model = base_model(is_gabornet=is_gabornet, kernel_size=kernel_size, add_padding=add_padding, 
                            gabor_type=gabor_type, device=device)
            model, _, model_epoch = load_net(checkpoint, model)

            assert epoch == model_epoch
            model_dict[epoch + 1] = model   # Epochs are 0-indexed in files, but we want 1-indexed.

    return gabor_models, cnn_models


def get_conv_weights(model) -> torch.Tensor:
    """Gets the weights of the first convolutional layer."""

    # Since the weights aren't updated directly (the parameters of the Gabor equation are), we need to update the
    # weights before returning them.
    if model.is_gabornet:
        return model.g1.calculate_weights()

    return model.g1.weight


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

    # TODO: deprecate.
    parser.add_argument("--gabor_type", type=str, default="GaborConv2d", help="Type of GaborNet to use. If specified in model save dict will use that. This is for backwards compatibility.", 
                        choices=["GaborConv2d", "GaborConv2dBuggy", "GaborConv2dStillBuggy"])
    
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
    gabor_models, cnn_models = load_models(base_model, epochs_to_load=epochs, 
                                           gabor_models_dir=gabor_models_dir, cnn_models_dir=cnn_models_dir, 
                                           gabor_type=globals()[args.gabor_type], device=device)

    # # Show the accuracies.
    # first_epoch = min(gabor_models.keys())
    # for epoch in gabor_models:
    #     if epoch == first_epoch:
    #         continue

    #     assert torch.allclose(get_conv_weights(gabor_models[first_epoch]), get_conv_weights(gabor_models[epoch]))

    # Data from the paper.
    gabor_train_reported = {1: 0.503, 3: 0.597, 10: 0.682, 40: 0.747} #, 90: 0.796}
    cnn_train_reported = {1: 0.506, 3: 0.520, 10: 0.613, 40: 0.674} #, 90: 0.732}
    gabor_test_reported = {1: 0.517, 3: 0.620, 10: 0.679, 40: 0.739} #, 90: 0.792}
    cnn_test_reported = {1: 0.503, 3: 0.515, 10: 0.616, 40: 0.668} #, 90: 0.726}

    # Recreate the figure from the original paper with our replication results.
    print(f"Calculating accuracies (based on N={args.N} samples)...")
    gabor_train, cnn_train, gabor_test, cnn_test = calc_accuracies(
        gabor_models, cnn_models, train_loader, test_loader, save_dir, use_cache=args.use_cache, device=device)

    print("Making figure...")
    make_accuracy_fig(gabor_train, cnn_train, gabor_test, cnn_test, 
             gabor_train_reported, cnn_train_reported, gabor_test_reported, cnn_test_reported, save_dir)


if __name__ == '__main__':
    main()
