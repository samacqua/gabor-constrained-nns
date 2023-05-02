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
from .dataset import DogsCatsDataset


def make_accuracy_fig(gabor_train: dict[int, float], cnn_train: dict[int, float], gabor_test: dict[int, float], 
             cnn_test: dict[int, float], gabor_train_reported: dict[int, float] = None, 
             cnn_train_reported: dict[int, float] = None, gabor_test_reported: dict[int, float] = None,
             cnn_test_reported: dict[int, float] = None) -> None:
    """Makes a figure of accuracy in the same way as the original paper."""

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 7))

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

    plt.show()


def calc_accuracies(gabor_models: dict[int, torch.nn.Module], cnn_models: dict[int, torch.nn.Module], 
               train_loader: DataLoader, test_loader: DataLoader, save_dir: str, 
               use_cache: bool = True) -> tuple[dict[int, float], dict[int, float], dict[int, float], dict[int, float]]:
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
        print("Calculating train accuracy...")
        gabor_train_accs = {}
        cnn_train_accs = {}
        for epoch in tqdm(epochs):
            gabor_model, cnn_model = gabor_models[epoch], cnn_models[epoch]
            gabor_model.eval()
            cnn_model.eval()
                
            # Calculate the accuracy over the train set.
            gabor_correct = 0
            cnn_correct = 0
            for data in train_loader:
                
                # Get the predictions.
                x, y = data["image"], data["target"]
                gabor_out = gabor_model(x)
                cnn_out = cnn_model(x)
                gabor_pred = gabor_out.argmax(dim=1)
                cnn_pred = cnn_out.argmax(dim=1)

                # Update the correct counts.
                gabor_correct += (gabor_pred == y).sum().item()
                cnn_correct += (cnn_pred == y).sum().item()

            # Calculate the accuracies.
            gabor_acc = gabor_correct / len(train_loader.dataset)
            cnn_acc = cnn_correct / len(train_loader.dataset)

            # Save the accuracies.
            gabor_train_accs[epoch] = gabor_acc
            cnn_train_accs[epoch] = cnn_acc

        # Calculate the accuracies over the test set.
        print("Calculating test accuracy...")
        gabor_test_accs = {}
        cnn_test_accs = {}

        for epoch in tqdm(epochs):
            gabor_model, cnn_model = gabor_models[epoch], cnn_models[epoch]
            gabor_model.eval()
            cnn_model.eval()
                    
            # Calculate the accuracy over the test set.
            gabor_correct = 0
            cnn_correct = 0
            for data in test_loader:

                # Get the predictions.
                x, y = data["image"], data["target"]
                gabor_out = gabor_model(x)
                cnn_out = cnn_model(x)
                gabor_pred = gabor_out.argmax(dim=1)
                cnn_pred = cnn_out.argmax(dim=1)

                # Update the correct counts.
                gabor_correct += (gabor_pred == y).sum().item()
                cnn_correct += (cnn_pred == y).sum().item()

            # Calculate the accuracies.
            gabor_acc = gabor_correct / len(test_loader.dataset)
            cnn_acc = cnn_correct / len(test_loader.dataset)

            # Save the accuracies.
            gabor_test_accs[epoch] = gabor_acc
            cnn_test_accs[epoch] = cnn_acc

        # Save the accuracies.
        with open(save_path, "wb") as f:
            pickle.dump((gabor_train_accs, cnn_train_accs, gabor_test_accs, cnn_test_accs), f)

    return gabor_train_accs, cnn_train_accs, gabor_test_accs, cnn_test_accs


def load_models(model_dir: str, base_model: torch.nn.Module = Type[torch.nn.Module], epochs_to_load: set[int] = None
                ) -> tuple[dict[int, torch.nn.Module], dict[int, torch.nn.Module]]:
    """Loads the Gabor + CNN models."""

    gabor_models = {}
    cnn_models = {}

    # Load the models by epoch.
    for fname in os.listdir(model_dir):
        if "epoch" in fname:
            epoch = int(fname.split("epoch_")[-1].split(".")[0])    # Files of format "{mdodel_name}_epoch_{epoch}.pth"

            # Only load those from the specified epochs.
            if epochs_to_load is not None and epoch not in epochs_to_load:
                continue

            is_gabornet = "gabor" in fname
            model_dict = gabor_models if is_gabornet else cnn_models
            checkpoint = torch.load(os.path.join(model_dir, fname))

            kernel_size = checkpoint.get("kernel_size", (15, 15) if is_gabornet else (5, 5))
            add_padding = checkpoint.get("add_padding", False)
            model = base_model(is_gabornet=is_gabornet, kernel_size=kernel_size, add_padding=add_padding)
            model, _, model_epoch = load_net(checkpoint, model)

            assert epoch == model_epoch
            model_dict[epoch + 1] = model   # Epochs are 0-indexed in files, but we want 1-indexed.

    return gabor_models, cnn_models


def main():

    # Parse the arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("seed", type=int, help="The random seed of the experiment to analyze.")
    
    parser.add_argument("--use_cache", action="store_true", help="If True, then if accuracies are already calculated, "
                                                                    "they will be loaded from cache.")
    parser.add_argument("--N", type=int, default=1024, help="The number of samples to use from the train / test sets.")
    parser.add_argument("--calc_every", type=int, default=3, help="The number of epochs to skip between calculating "
                                                                     "the accuracies.")
    parser.add_argument("--base_model", type=str, default="DogCatNet", help="The base model to use for the CNNs.")
    args = parser.parse_args()

    # Load the training arguments.
    save_dir = f"recreate/out/seed_{args.seed}/"
    with open(os.path.join(save_dir, "args.json"), "rb") as f:
        train_args = json.load(f)

    base_model = train_args['model']
    if base_model == "DogCatNet":
        base_model = DogCatNet
    elif base_model == "DogCatNNSanity":
        base_model = DogCatNNSanity
    else:
        raise ValueError(f"Invalid base model: {base_model}")
    
    # Load the testset.
    print("Loading dataset...")
    torch.manual_seed(args.seed)
    train_set, test_set = load_dataset(train_args['dataset_dir'], img_size=train_args['img_size'])

    # Split the trainset into train and test.
    train_set, test_set1 = torch.utils.data.random_split(
        train_set, [int(len(train_set) * 0.7), int(len(train_set) * 0.3)]
    )

    # Limit the dataset size to make the computations faster.
    train_set, _ = torch.utils.data.random_split(train_set, [args.N, len(train_set) - args.N])
    val_set, _ = torch.utils.data.random_split(test_set1, [args.N, len(test_set1) - args.N])
    test_set, _ = torch.utils.data.random_split(test_set, [args.N, len(test_set) - args.N])
    test_set = val_set      # Pretty sure the original papers "test set" is actually a validation set.

    batch_size = 64
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=0)

    # Load the models.
    print("Loading models...")
    epochs = range(0, 100, args.calc_every)
    gabor_models, cnn_models = load_models(save_dir, base_model, epochs_to_load=epochs)

    # Data from the paper.
    gabor_train_reported = {1: 0.503, 3: 0.597, 10: 0.682, 40: 0.747} #, 90: 0.796}
    cnn_train_reported = {1: 0.506, 3: 0.520, 10: 0.613, 40: 0.674} #, 90: 0.732}
    gabor_test_reported = {1: 0.517, 3: 0.620, 10: 0.679, 40: 0.739} #, 90: 0.792}
    cnn_test_reported = {1: 0.503, 3: 0.515, 10: 0.616, 40: 0.668} #, 90: 0.726}

    # Recreate the figure from the original paper with our replication results.
    print(f"Calculating accuracies (based on N={args.N} samples)...")
    gabor_train, cnn_train, gabor_test, cnn_test = calc_accuracies(
        gabor_models, cnn_models, train_loader, test_loader, save_dir, use_cache=args.use_cache)

    print("Making figure...")
    make_accuracy_fig(gabor_train, cnn_train, gabor_test, cnn_test, 
             gabor_train_reported, cnn_train_reported, gabor_test_reported, cnn_test_reported)


if __name__ == '__main__':
    main()
