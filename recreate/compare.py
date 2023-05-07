"""Script to make general comparisons between models, loads from compare.json."""

import matplotlib.pyplot as plt
import os
import argparse
import torch
import json

from torch.utils.data import DataLoader

from .analyze import load_models, calc_accuracies_full
from .dogs_cats import DogCatNNSanity, DogCatNet, load_dataset


def compare_models(models_train: dict[str, dict[int, float]], models_test: dict[str, dict[int, float]], save_dir: str):
    """Makes plots to compare the performance of many models."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # Plot the training accuracy.
    for model_name, model_accs in models_train.items():
        train_x, train_y = zip(*sorted(model_accs.items()))
        ax1.plot(train_x, train_y, label=model_name)

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training Accuracy")
    ax1.set_title("Training Accuracy")
    ax1.legend()

    # Plot the test accuracy.
    for model_name, model_accs in models_test.items():
        test_x, test_y = zip(*sorted(model_accs.items()))
        ax2.plot(test_x, test_y, label=model_name)

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

    parser.add_argument("--seed", type=int, help="The random seed of the experiment to analyze.")

    parser.add_argument("--model", type=str, default="DogCatNet", help="The base model to use for the CNNs.",
                        choices=["DogCatNet", "DogCatNNSanity"])
    
    parser.add_argument("--epochs", type=int, nargs="+", default=None, help="1-index epochs to calc accuracies for.")
    parser.add_argument("--use_cache", action="store_true", help="If True, then if accuracies are already calculated, "
                                                                    "they will be loaded from cache.")
    parser.add_argument("--dataset_dir", type=str, default=None, help="Path to dataset.")
    parser.add_argument("--N", type=int, default=None, help="The number of samples to use from the train / test sets.")
    
    
    args = parser.parse_args()

    base_model = globals()[args.model]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open("recreate/compare.json", "r") as f:
        model_dirs = json.load(f)

    # Ensure that models are comparable + set dataset path.
    img_size = None
    dataset_dir = args.dataset_dir
    for model_name, model_dir in model_dirs.items():
        exp_path = get_exp_path(model_dir)
        exp_args_path = os.path.join(exp_path, "args.json")
        with open(exp_args_path, "r") as f:
            exp_args = json.load(f)

        if img_size is None:
            img_size = exp_args['img_size']
        else:
            assert img_size == exp_args['img_size'], "Image sizes are not the same."

        if not args.dataset_dir:
            if dataset_dir is None:
                dataset_dir = exp_args['dataset_dir']
            else:
                assert dataset_dir == exp_args['dataset_dir'], "Dataset directories are not the same."


    # Load the models.
    models = {}
    for model_name, model_dir in model_dirs.items():
        accuracy_dir = get_acc_path(model_dir)
        models[model_name] = {
            "checkpoints": load_models(base_model, args.epochs, model_dir, device, calc_weights=False),
            "save_dir": accuracy_dir
            }
        
    # Load the testset.
    print("Loading dataset...")
    torch.manual_seed(args.seed)
    train_set, test_set = load_dataset(dataset_dir, img_size=(img_size, img_size))

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
    
    # Get the training + testing accuracies.
    print("Calculating accuracies...")
    model_acc_dirs = calc_accuracies_full(models, train_loader, test_loader, use_cache=args.use_cache, device=device)

    # {model_name: ({epoch: train_accuracy}, {epoch: test_accuracy}})
    train_accs = {name: accs[0] for name, accs in model_acc_dirs.items()}
    test_accs = {name: accs[1] for name, accs in model_acc_dirs.items()}

    # Compare the models.
    print("Plotting accuracies...")
    compare_models(train_accs, test_accs, "./")


if __name__ == "__main__":
    main()
