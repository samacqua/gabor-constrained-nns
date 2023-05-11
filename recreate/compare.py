"""Script to make general comparisons between models, loads from compare.json."""

import matplotlib.pyplot as plt
import os
import argparse
import torch
import json
import numpy as np

from torch.utils.data import DataLoader

from .analyze import load_models, calc_accuracies_full
from .dogs_cats import DogCatNet
from src.datasets import load_dataset
from src.models import CNN, CNNSmall, CNNLinear


def compare_models(models_train: dict[str, dict[int, float | tuple[float, float]]], 
                   models_test: dict[str, dict[int, float | tuple[float, float]]], save_dir: str):
    """Makes plots to compare the performance of many models."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the training accuracy.
    for model_name, model_accs in models_train.items():
        train_x, train_y = zip(*sorted(model_accs.items()))
        train_y, std_err = zip(*train_y)
        p = ax1.plot(train_x, train_y, label=model_name)
        ax1.errorbar(train_x, train_y, yerr=std_err, fmt="o", capsize=5, color=p[0].get_color())

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Accuracy")
    ax1.set_title("Training Accuracy")
    ax1.legend()

    # Plot the test accuracy.
    for model_name, model_accs in models_test.items():
        test_x, test_y = zip(*sorted(model_accs.items()))
        test_y, std_err = zip(*test_y)
        p = ax2.plot(test_x, test_y, label=model_name)
        ax2.errorbar(test_x, test_y, yerr=std_err, fmt="o", capsize=5, color=p[0].get_color())

    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test Accuracy")
    ax2.set_title("Test Accuracy")
    ax2.legend()

    plt.savefig(os.path.join(save_dir, "model_comparison.png"))
    plt.show()


def get_exp_path(model_dir: str) -> str:
    """Gets the path to the experiment directory from a model directory path."""
    # model_dir has form {path}/{exp_name}/models/{model_name}/
    all_models_dir, _ = os.path.split(os.path.normpath(model_dir))     # {path}/{exp_name}/models/
    exp_dir, _ = os.path.split(os.path.normpath(all_models_dir))     # {path}/{exp_name}/
    return exp_dir


def get_acc_path(model_dir: str) -> str:
    """Gets the path to the accuracies directory from a model directory path."""
    model_name = os.path.split(os.path.normpath(model_dir))[-1]
    return os.path.join(get_exp_path(model_dir), "accuracy", model_name)


def main():

    parser = argparse.ArgumentParser()
    
    # Analysis params.
    parser.add_argument("--epochs", type=int, nargs="+", default=None, help="1-index epochs to calc accuracies for.")
    parser.add_argument("--use_cache", action="store_true", help="If True, then if accuracies are already calculated, "
                                                                    "they will be loaded from cache.")
    # Dataset params.
    parser.add_argument("--dataset_dir", type=str, default="data/dogs-vs-cats/", help="Path to dataset.")
    parser.add_argument("--N", type=int, default=None, help="The number of samples to use from the train / test sets.")

    
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the comparison json file.
    with open("recreate/compare.json", "r") as f:
        model_dirs_ = json.load(f)

        # Convert seed string to int.
        model_dirs = {}
        for model_name in model_dirs_:
            for seed_str, dir in model_dirs_[model_name].items():
                model_dirs.setdefault(model_name, {})[int(seed_str)] = dir

    # Ensure that models are comparable + set dataset path.
    img_size, n_channels, dataset, base_model = None, None, None, None
    dataset_dir = args.dataset_dir
    for model_name, model_dir_by_seed in model_dirs.items():
        exp_path = get_exp_path(list(model_dir_by_seed.values())[0])    # All have the same experiment info.

        exp_args_path = os.path.join(exp_path, "args.json")
        with open(exp_args_path, "r") as f:
            exp_args = json.load(f)

        assert img_size == exp_args['img_size'] or img_size is None, "Image sizes are not the same."
        img_size = exp_args['img_size']

        if not args.dataset_dir:
            assert dataset_dir == exp_args['dataset_dir'] or dataset_dir is None, "Dataset directories are not the same."
            dataset_dir = exp_args['dataset_dir']

        assert base_model == exp_args['model'] or base_model is None, "Base models are not the same."
        base_model = exp_args['model']

        assert dataset == exp_args['dataset'] or dataset is None, "Datasets are not the same."
        dataset = exp_args['dataset']
        
        assert n_channels == exp_args['n_channels'] or n_channels is None, "Number of channels are not the same."
        n_channels = exp_args['n_channels'] 

    base_model = globals()[base_model]
    epochs = [epoch - 1 for epoch in args.epochs]   # 0-index epochs.

    # Load the models.
    models_by_seed = {}
    for model_name, model_dir_by_seed in model_dirs.items():
        for seed, model_dir in model_dir_by_seed.items():
            accuracy_dir = get_acc_path(model_dir)
            models_by_seed.setdefault(seed, {})[model_name] = {
                "checkpoints": load_models(base_model, epochs, model_dir, device),
                "save_dir": accuracy_dir
                }

    train_accs_by_seed = {}
    test_accs_by_seed = {}
    for seed, models in models_by_seed.items():

        # Load the testset.
        print(f"Loading dataset with seed={seed}...")
        torch.manual_seed(seed)
        train_set, test_set = load_dataset(dataset, dataset_dir=dataset_dir, img_size=img_size, n_channels=n_channels)

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
        # test_set = val_set      # Pretty sure the original papers "test set" is actually a validation set.

        batch_size = 64
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=0)
        
        # Get the training + testing accuracies.
        print("Calculating accuracies...")
        model_acc_dirs = calc_accuracies_full(models, train_loader, test_loader, use_cache=args.use_cache, device=device)

        # {model_name: ({epoch: train_accuracy}, {epoch: test_accuracy}})
        train_accs = {name: accs[0] for name, accs in model_acc_dirs.items()}
        test_accs = {name: accs[1] for name, accs in model_acc_dirs.items()}
        train_accs_by_seed[seed] = train_accs
        test_accs_by_seed[seed] = test_accs

    # Get the mean + standard error across seeds.
    train_accs = {}
    test_accs = {}
    for seed, train_accs_ in train_accs_by_seed.items():
        for model_name, accs in train_accs_.items():
            train_accs.setdefault(model_name, []).append(accs)
    for seed, test_accs_ in test_accs_by_seed.items():
        for model_name, accs in test_accs_.items():
            test_accs.setdefault(model_name, []).append(accs)

    train_acc_stats = {}
    test_acc_stats = {}
    for model_name, accs in train_accs.items():
        model_stats = {}
        for epoch in args.epochs:
            model_stats[epoch] = (np.mean([acc[epoch] for acc in accs]), np.std([acc[epoch] for acc in accs]) / np.sqrt(len(accs)))

        train_acc_stats[model_name] = model_stats

    for model_name, accs in test_accs.items():
        model_stats = {}
        for epoch in args.epochs:
            model_stats[epoch] = (np.mean([acc[epoch] for acc in accs]), np.std([acc[epoch] for acc in accs]) / np.sqrt(len(accs)))

        test_acc_stats[model_name] = model_stats

    # Compare the models.
    print("Plotting accuracies...")
    compare_models(train_acc_stats, test_acc_stats, "./")


if __name__ == "__main__":
    main()
