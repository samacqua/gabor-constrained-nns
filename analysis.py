"""Analysis of the results of the experiments + code to generate figures for final paper.

You must point to an experiment that has already been run and has a "cnn" schedule and a "gabor" schedule.
"""

import argparse
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import pickle
import scipy.stats as stats
import json

import torch
from torchvision import utils
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchattacks import PGD, FGSM

from parse_config import parse_config

import warnings
warnings.filterwarnings("ignore")


def test_generalization_hypothesis(models: list[dict[str, dict[int, torch.nn.Module]]], test_loader_b, device, 
                                   save_dir: str, use_cache: bool = True):
    """Tests the hypothesis: Gabor-constrained neural networks will finetune better (converge faster + higher acc).
    
    Args:
        models: list of models finetuned on b, where each model dict is of the form
            {model_name: {"checkpoints":{epoch: model}, "save_dir": save_dir}} and is a repeat.
        unconstrained_models: A dictionary of list of CNN model checkpoints, indexed by epoch.
        test_loader_b: The dataloader for the test set of dataset B.
        device: The device to run the models on.
    """

    assert {"gabor", "cnn", "baseline"} == set(models[0]), \
        f"Model sequence must be 'gabor', 'cnn', and 'baseline', not {set(models[0])}."

    epochs = sorted(list(models[0]["gabor"]["checkpoints"].keys()))

    # Get the accuracy of each model at each stage of convergence.
    gabor_accuracies, unconstrained_accuracies, baseline_accuracies = [], [], []
    for repeat_models in models:
        accuracies = calc_accuracies_full(repeat_models, test_loader=test_loader_b, device=device, use_cache=use_cache)

        gabor_accuracies.append({epoch: accuracies["gabor"][1][epoch] for epoch in epochs})
        unconstrained_accuracies.append({epoch: accuracies["cnn"][1][epoch] for epoch in epochs})
        baseline_accuracies.append({epoch: accuracies["baseline"][1][epoch] for epoch in epochs})

    # TODO: Check if the difference is statistically significant.
    pass

    # Calculate the mean accuracy of each model at each stage of convergence.
    N = len(models)
    gabor_mean_accuracies = [np.mean([gabor_accuracies[i][epoch] for i in range(N)]) for epoch in epochs]
    gabor_conf = np.array([np.std([gabor_accuracies[i][epoch] for i in range(N)]) / np.sqrt(N) for epoch in epochs]) * 1.96
    unconstrained_mean_accuracies = [np.mean([unconstrained_accuracies[i][epoch] for i in range(N)]) for epoch in epochs]
    unconstrained_conf = np.array([np.std([unconstrained_accuracies[i][epoch] for i in range(N)]) / np.sqrt(N) for epoch in epochs]) * 1.96
    baseline_mean_accuracies = [np.mean([baseline_accuracies[i][epoch] for i in range(N)]) for epoch in epochs]
    baseline_conf = np.array([np.std([baseline_accuracies[i][epoch] for i in range(N)]) / np.sqrt(N) for epoch in epochs]) * 1.96

    # Plot the accuracy of each model at each stage of convergence.
    fig, ax = plt.subplots()

    x = range(len(gabor_mean_accuracies))
    ax.errorbar(x, gabor_mean_accuracies, yerr=gabor_conf, label="Gabor")
    ax.errorbar(x, unconstrained_mean_accuracies, yerr=unconstrained_conf, label="Unconstrained")
    ax.errorbar(x, baseline_mean_accuracies, yerr=baseline_conf, label="Baseline")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy of Gabor and Unconstrained Models at Each Stage of Convergence")
    ax.legend()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.png"))
    plt.show()


def test_plasticity_hypothesis(models, test_loader_a, 
                               test_loader_b, device, save_dir, use_cache: bool = True):
    """Tests the hypothesis: Gabor-constrained neural networks will retain original performance better.

    Expects that the models are finetuned to the same accuracy or loss.
    
    Args:
        models: list of models finetuned on b {model_name: {"checkpoints":{epoch: model}, "save_dir": save_dir}}.
        test_loader_a: The dataloader for the test set of dataset A.
        test_loader_b: The dataloader for the test set of dataset B.
        device: The device to run the models on.
    """

    # TODO: Check that actually tuned to same loss.

    assert {"gabor", "cnn", "baseline"} == set(models[0]), \
        f"Model sequence must be 'gabor', 'cnn', and 'baseline', not {set(models[0])}."

    epochs = sorted(list(models[0]["gabor"]["checkpoints"].keys()))

    # Calculate accuracies.
    gabor_accuracies, unconstrained_accuracies, baseline_accuracies = [], [], []
    for repeat_models in models:

        accuracies_b = calc_accuracies_full(repeat_models, test_loader=test_loader_b, device=device, use_cache=use_cache)
        accuracies_a = calc_accuracies_full(repeat_models, test_loader=test_loader_a, device=device, use_cache=use_cache)

        gabor_accuracies.append([(accuracies_a["gabor"][1][epoch], accuracies_b["gabor"][1][epoch]) for epoch in epochs])
        unconstrained_accuracies.append([(accuracies_a["cnn"][1][epoch], accuracies_b["cnn"][1][epoch]) for epoch in epochs])
        baseline_accuracies.append([(accuracies_a["baseline"][1][epoch], accuracies_b["baseline"][1][epoch]) for epoch in epochs])

    # TODO: Check if the difference is statistically significant.
    pass

    # Create mapping from accuracy on dataset A to accuracy on dataset B.
    gabor_accuracies_ba = [item for sublist in gabor_accuracies for item in sublist]
    unconstrained_accuracies_ba = [item for sublist in unconstrained_accuracies for item in sublist]
    baseline_accuracies_ba = [item for sublist in baseline_accuracies for item in sublist]

    # Calculate the mean accuracy of each model on dataset A at each stage of convergence of training on B.
    N = len(models)
    gabor_mean_accuracies_a = [np.mean([gabor_accuracies[i][epoch][0] for i in range(N)]) for epoch in epochs]
    gabor_conf = np.array([np.std([gabor_accuracies[i][epoch][0] for i in range(N)]) / np.sqrt(N) for epoch in epochs]) * 1.96
    unconstrained_mean_accuracies_a = [np.mean([unconstrained_accuracies[i][epoch][0] for i in range(N)]) for epoch in epochs]
    unconstrained_conf = np.array([np.std([unconstrained_accuracies[i][epoch][0] for i in range(N)]) / np.sqrt(N) for epoch in epochs]) * 1.96
    baseline_mean_accuracies_a = [np.mean([baseline_accuracies[i][epoch][0] for i in range(N)]) for epoch in epochs]
    baseline_conf = np.array([np.std([baseline_accuracies[i][epoch][0] for i in range(N)]) / np.sqrt(N) for epoch in epochs]) * 1.96

    # Plot the accuracy of each model at each stage of convergence.
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 5))
    x = range(len(gabor_mean_accuracies_a))
    ax0.errorbar(x, gabor_mean_accuracies_a, yerr=gabor_conf, label="Gabor")
    ax0.errorbar(x, unconstrained_mean_accuracies_a, yerr=unconstrained_conf, label="Unconstrained")
    ax0.errorbar(x, baseline_mean_accuracies_a, yerr=baseline_conf, label="Baseline")

    ax0.set_xlabel("epoch")
    ax0.set_ylabel("Accuracy on Dataset A")
    ax0.legend()

    ax1.scatter([b for a, b in gabor_accuracies_ba], [a for a, b in gabor_accuracies_ba], label="Gabor")
    ax1.scatter([b for a, b in unconstrained_accuracies_ba], [a for a, b in unconstrained_accuracies_ba], label="Unconstrained")
    ax1.scatter([b for a, b in baseline_accuracies_ba], [a for a, b in baseline_accuracies_ba], label="Baseline")
    ax1.set_xlabel("Accuracy on Dataset B")
    ax1.set_ylabel("Accuracy on Dataset A")
    ax1.legend()

    fig.suptitle("Accuracy of Gabor and Unconstrained Models at Each Stage of Convergence")

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "accuracy_vs_accuracy.png"))
    plt.show()


def test_adversarial_hypothesis(models, test_loader, device, save_dir, epsilon=0.1):
    """Tests the hypothesis: Gabor-constrained neural networks will be more robust to adversarial attacks.

    Expects that the models are finetuned to the same accuracy or loss.
    
    Args:
        models: list of models finetuned on b {model_name: {"checkpoints":{epoch: model}, "save_dir": save_dir}}.
        test_loader: The dataloader for the test set.
        device: The device to run the models on.
    """

    cnn_acc, gabor_acc, cnn_adv_acc, gabor_adv_acc = [], [], [], []
    models_repeats = models

    with tqdm(total=len(test_loader) * 6 * len(models), desc=f"With epsilon={epsilon}") as pbar:
        for models in models_repeats:

            # Get the CNN + Gabor model.
            cnn_max_epoch = sorted(list(models["cnn"]["checkpoints"].keys()))[-1]
            cnn_model = models["cnn"]["checkpoints"][cnn_max_epoch]
            gabor_max_epoch = sorted(list(models["gabor"]["checkpoints"].keys()))[-1]
            gabor_model = models["gabor"]["checkpoints"][gabor_max_epoch]

            # Calculate the accuracy of each model on the test set.
            cnn_acc.append(test_accuracy(test_loader=test_loader, model=cnn_model, device=device, pbar=pbar))
            gabor_acc.append(test_accuracy(test_loader=test_loader, model=gabor_model, device=device, pbar=pbar))

            # Set up the adversarial attack.
            batch_size, n_channels, _, _ = next(iter(test_loader))[0].shape
            cnn_atk = PGD(cnn_model, eps=epsilon, alpha=2/225, steps=10, random_start=True)
            cnn_atk.set_normalization_used(mean=(0.5,) * n_channels, std=(0.5,) * n_channels)
            gabor_atk = PGD(gabor_model, eps=epsilon, alpha=2/225, steps=10, random_start=True)
            gabor_atk.set_normalization_used(mean=(0.5,) * n_channels, std=(0.5,) * n_channels)

            # Make a dataset of the adversarial examples.
            cnn_adv_examples = []
            gabor_adv_examples = []
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                cnn_adv_examples.append(cnn_atk(data, target))
                pbar.update(1)
                gabor_adv_examples.append(gabor_atk(data, target))
                pbar.update(1)

            # Make a dataloader of the adversarial examples.
            cnn_adv_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.cat(cnn_adv_examples), torch.cat([target for _, target in test_loader])),
                batch_size=batch_size)
            gabor_adv_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(torch.cat(gabor_adv_examples), torch.cat([target for _, target in test_loader])),
                batch_size=batch_size)
            
            # Calculate the accuracy of each model on the adversarial examples.
            cnn_adv_acc.append(test_accuracy(test_loader=cnn_adv_loader, model=cnn_model, device=device, pbar=pbar))
            gabor_adv_acc.append(test_accuracy(test_loader=gabor_adv_loader, model=gabor_model, device=device, pbar=pbar))

    # Print the results.
    cnn_acc_mean, cnn_acc_conf = np.mean(cnn_acc), 1.96 * np.std(cnn_acc) / np.sqrt(len(cnn_acc))
    gabor_acc_mean, gabor_acc_conf = np.mean(gabor_acc), 1.96 * np.std(gabor_acc) / np.sqrt(len(gabor_acc))
    cnn_adv_acc_mean, cnn_adv_acc_conf = np.mean(cnn_adv_acc), 1.96 * np.std(cnn_adv_acc) / np.sqrt(len(cnn_adv_acc))
    gabor_adv_acc_mean, gabor_adv_acc_conf = np.mean(gabor_adv_acc), 1.96 * np.std(gabor_adv_acc) / np.sqrt(len(gabor_adv_acc))
    print(f"Accuracy on test set: CNN: {cnn_acc_mean:.5f} ± {cnn_acc_conf:.5f}, Gabor: {gabor_acc_mean:.5f} ± {gabor_acc_conf:.5f}")
    print(f"Accuracy on adversarial examples: CNN: {cnn_adv_acc_mean:.5f} ± {cnn_adv_acc_conf:.5f}, Gabor: {gabor_adv_acc_mean:.5f} ± {gabor_adv_acc_conf:.5f}")


def visualize_tensor(tensor: torch.Tensor, ch: int = 0, allkernels: bool = False, nrow: int = 8, padding: int = 1, 
                     save_dir: str = None, title: str = None): 
    """Visualizes a tensor."""
    n,c,w,h = tensor.shape

    if allkernels: 
        tensor = tensor.view(n*c, -1, w, h)
    elif c != 3: 
        tensor = tensor[:,ch,:,:].unsqueeze(dim=1)

    rows = np.min((tensor.shape[0] // nrow + 1, 64))    
    grid = utils.make_grid(tensor, nrow=nrow, normalize=True, padding=padding)
    plt.figure( figsize=(nrow,rows) )
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    if title:
        plt.title(title)

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "features.png"))

    plt.show()


def visualize_features(model: torch.nn.Module, save_dir: str, title: str):
    """Visualizes the features of a model."""

    # Get the weights of the first layer.
    weights = model.g1.weight.data.cpu()
    visualize_tensor(weights, ch=0, allkernels=False, save_dir=save_dir, title=title)


def test_accuracy(test_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device, pbar: tqdm = None):
    """Tests the accuracy of a model."""
    
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            if pbar is not None:
                pbar.update(1)
    
    return correct / total


def load_model(base_model: torch.nn.Module, is_gabornet: bool, n_channels: int, save_path: str):
    """Loads a model."""

    model = base_model      # (is_gabornet=is_gabornet, n_channels=n_channels)
    try:
        model.load_state_dict(torch.load(save_path)["model_state_dict"])
    except FileNotFoundError:
        print("Could not load model from path: {}".format(save_path))
        model = None

    return model


def load_models(config: dict, intermediate: bool = False) -> tuple[dict[str, torch.nn.Module], dict[str, torch.nn.Module]]:
    """Loads the final models."""

    a_epochs = config["training"]["initial"]["epochs"]
    b_epochs = config["training"]["finetune"]["epochs"]

    a_model_save_dir = os.path.join(config['save_dir'], "dataset_a", "models")
    b_model_save_dir = os.path.join(config['save_dir'], "dataset_b", "models")
    base_model = config['base_model']
    n_channels = config['datasets']['params']['n_channels']

    a_models = sorted(os.listdir(a_model_save_dir))
    b_models = sorted(os.listdir(b_model_save_dir))
    model_as = {}
    model_bs = {}
    for model_sequence, model_sequence_ in zip(a_models, b_models):

        assert model_sequence == model_sequence_, "Model sequences do not match between the 2 datasets."
        assert model_sequence in ['gabor', 'cnn', 'baseline'], "Model sequence not supported."

        # Load the models from dataset A.
        init_cfg = config['schedules'][model_sequence]['initial_train']
        base_model = config['schedules'][model_sequence]['model']
        model_a_checkpoints = {}
        for i in range(0 if intermediate else a_epochs - 1, a_epochs):
            model_a_path = os.path.join(a_model_save_dir, model_sequence, f"epoch_{i}.pth")
            model_a_checkpoints[i] = load_model(deepcopy(base_model), init_cfg['gabor_constrained'], n_channels, model_a_path)

        # Load the models from dataset B.
        finetune_cfg = config['schedules'][model_sequence]['finetune']
        model_b_checkpoints = {}
        for i in range(0 if intermediate else b_epochs - 1, b_epochs):
            model_b_path = os.path.join(b_model_save_dir, model_sequence, f"epoch_{i}.pth")
            model_b_checkpoints[i] = load_model(deepcopy(base_model), finetune_cfg['gabor_constrained'], n_channels, model_b_path)

        model_as[model_sequence] = {
            "checkpoints": model_a_checkpoints, 
            "save_dir": os.path.join(config['save_dir'], "dataset_a", "accuracy", model_sequence)
        }
        model_bs[model_sequence] = {
            "checkpoints": model_b_checkpoints, 
            "save_dir": os.path.join(config['save_dir'], "dataset_b", "accuracy", model_sequence)
        }

    return model_as, model_bs


def calc_accuracies(models: dict[int, torch.nn.Module], dataloader: DataLoader, cache_path: str = None, 
                    device: str = "cpu", pbar=None, use_cache: bool = False) -> dict[int, float]:
    """Calculates the accuracy of a single model over the course of training on 1 dataloader."""

    accuracies = {}
    pbar = tqdm(total=len(models) * len(dataloader)) if pbar is None else pbar

    dataset_name = dataloader.dataset.__class__.__name__
    dataset_name = dataloader.dataset.dataset.__class__.__name__ if dataset_name == "Subset" else dataset_name

    # Load from cache if it exists and was calculated with at least as many samples as the current request.
    accuracy_dict = {}
    if cache_path and os.path.exists(cache_path) and use_cache:
        assert cache_path.endswith(".json"), "Cache path must be a JSON file."
        with open(cache_path, "r") as f:
            accuracy_dict = json.load(f)

        if "n_samples" in accuracy_dict:    # Backwards compatibile before indexing by dataset.
            accuracy_dict = {dataset_name: accuracy_dict}

        n_samples = accuracy_dict.get(dataset_name, {}).get("n_samples", 0)
        if n_samples >= len(dataloader.dataset):
            accuracies = {int(k): v for k, v in accuracy_dict[dataset_name]["accuracies"].items() if int(k) in models}

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
            accuracy_dict[dataset_name] = {"accuracies": accuracies, "n_samples": len(dataloader.dataset)}
            json.dump(accuracy_dict, f)

    return accuracies


def calc_accuracies_full(models: dict[str, dict[int, torch.nn.Module]],
               train_loader: DataLoader = None, test_loader: DataLoader = None, 
               use_cache: bool = True, device: str = 'cpu'
               ) -> dict[str, tuple[dict[int, float], dict[int, float]]]:
    """Calculates the accuracies over epochs on the train / test datasets for the multiple models.
    
    Args:
        models: {model_name: {"checkpoints":{epoch: model}, "save_dir": save_dir}}
        train_loader: The dataloader for the training dataset.
        test_loader: The dataloader for the test dataset.
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
    n_iters = n_checkpoints * ((len(train_loader) if train_loader else 0) + (len(test_loader) if test_loader else 0))

    model_accuracies = {}
    with tqdm(total=n_iters, maxinterval=5) as pbar:
            
        # Calculate the accuracies for each model.
        for model_name, model_info in models.items():

            # Load the checkpoints.
            checkpoints = model_info['checkpoints']
            save_dir = model_info['save_dir']

            # Calculate the accuracies.
            train_accuracies = calc_accuracies(checkpoints, train_loader, cache_path=os.path.join(save_dir, "train_accuracies.json"), 
                                            device=device, pbar=pbar, use_cache=use_cache) if train_loader else None
            test_accuracies = calc_accuracies(checkpoints, test_loader, cache_path=os.path.join(save_dir, "test_accuracies.json"), 
                                            device=device, pbar=pbar, use_cache=use_cache) if test_loader else None

            # Save the accuracies.
            model_accuracies[model_name] = (train_accuracies, test_accuracies)

    return model_accuracies


def main():

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    parser.add_argument("--no_cache", action="store_true", help="Don't use the cache.")
    parser.add_argument("--all", action="store_true", help="Run all the analyses.")

    parser.add_argument("--test", action="store_true", help="Calculate test accuracy of the models.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the features of the models.")
    parser.add_argument("--accuracy_a", action="store_true", help="Run analysis on the accuracy on dataset A.")
    parser.add_argument("--generalization", action="store_true", help="Run the generalization analysis.")
    parser.add_argument("--plasticity", action="store_true", help="Run the plasticity analysis.")
    parser.add_argument("--adversarial", action="store_true", help="Run the adversarial analysis.")

    args = parser.parse_args()

    # Parse the configuration file.
    config = parse_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cache = not args.no_cache

    # Load the models for each repeat.
    og_save_dir = config['save_dir']
    models_a_repeats = []
    models_b_repeats = []
    for repeat in range(config['n_repeats']):
        config['save_dir'] = os.path.join(og_save_dir, str(repeat))

        # Create the analysis folder.
        analysis_dir = os.path.join(config['save_dir'], "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # Load the models.
        models_a, models_b = load_models(config, intermediate=(
            args.all or args.accuracy_a or args.generalization or args.plasticity))
        models_a_repeats.append(models_a)
        models_b_repeats.append(models_b)

    # Load the datasets.
    if args.all or args.accuracy_a or args.generalization or args.plasticity or args.test or args.adversarial:
        print("Loading datasets...")
        testloader_a = torch.utils.data.DataLoader(config['datasets']['initial'][1], batch_size=128, shuffle=False)
        testloader_b = torch.utils.data.DataLoader(config['datasets']['finetune'][1], batch_size=128, shuffle=False)

    # Test the final accuracies of the models.
    if args.all or args.test:
        accuracies_a = []
        accuracies_b = []
        accuracies_ba = []
        for repeat, (models_a, models_b) in enumerate(zip(models_a_repeats, models_b_repeats)):
            print(f"\nRepeat {repeat}:")

            print("Testing accuracies on dataset A...")
            acc_a = calc_accuracies_full(models_a, test_loader=testloader_a, device=device, use_cache=use_cache)
            acc_a = {model_sequence: max(accs[1].items())[1] for model_sequence, accs in acc_a.items()}
            accuracies_a.append(acc_a)

            print("Testing accuracies on dataset B...")
            acc_b = calc_accuracies_full(models_b, test_loader=testloader_b, device=device, use_cache=use_cache)
            acc_b = {model_sequence: max(accs[1].items())[1] for model_sequence, accs in acc_b.items()}
            accuracies_b.append(acc_b)

            print("Testing accuracies on dataset A after finetuning on B...")
            acc_ba = calc_accuracies_full(models_b, test_loader=testloader_a, device=device, use_cache=use_cache)
            acc_ba = {model_sequence: max(accs[1].items())[1] for model_sequence, accs in acc_ba.items()}
            accuracies_ba.append(acc_ba)

        # Print the results.
        max_name_len = max([len(name) for name in accuracies_a[0]])
        padded_name = "Name" + ' ' * (max_name_len - 4)
        print(f"{padded_name}\tA\t\t\tB\t\t\tA (trained on B)")
        for model_sequence in accuracies_a[0]:
            # Get accuracies after last epoch.
            acc_a = np.mean([acc[model_sequence] for acc in accuracies_a])
            a_std_err = np.std([acc[model_sequence] for acc in accuracies_a]) / np.sqrt(len(accuracies_a))
            acc_a_range = round(acc_a - a_std_err, 3), round(acc_a + a_std_err, 3)

            acc_b = np.mean([acc[model_sequence] for acc in accuracies_b])
            b_std_err = np.std([acc[model_sequence] for acc in accuracies_b]) / np.sqrt(len(accuracies_b))
            acc_b_range = round(acc_b - b_std_err, 3), round(acc_b + b_std_err, 3)

            acc_ba = np.mean([acc[model_sequence] for acc in accuracies_ba])
            ba_std_err = np.std([acc[model_sequence] for acc in accuracies_ba]) / np.sqrt(len(accuracies_ba))
            acc_ba_range = round(acc_ba - ba_std_err, 3), round(acc_ba + ba_std_err, 3)

            padded_name = model_sequence + ' ' * (max_name_len - len(model_sequence))
            print(f"{padded_name}\t{acc_a:.3f} {acc_a_range}\t{acc_b:.3f} {acc_b_range}\t{acc_ba:.3f} {acc_ba_range}")

    # Visualize the features of each model.
    if args.all or args.visualize:
        print("Visualizing features...")
        models_a, models_b = models_a_repeats[-1], models_b_repeats[-1]
        for model_sequence in models_a:
            # model_a = max(models_a[model_sequence]["checkpoints"].items())[1]   # Get the last model.
            # a_save_dir = models_a[model_sequence]["save_dir"]
            # visualize_features(model_a, save_dir=a_save_dir, title=model_sequence)

            model_b = max(models_b[model_sequence]["checkpoints"].items())[1]   # Get the last model.
            b_save_dir = models_b[model_sequence]["save_dir"]
            visualize_features(model_b, save_dir=b_save_dir, title=model_sequence)

    # Run the analysis of convergence on dataset A.
    if args.all or args.accuracy_a:
        print("Running accuracy analysis on dataset A...")
        test_generalization_hypothesis(models_a_repeats, testloader_a, device, 
                                       save_dir=os.path.join(analysis_dir, 'accuracy_a'), use_cache=use_cache)

    # Run the generalization analysis.
    if args.all or args.generalization:
        print("Running generalization analysis...")
        test_generalization_hypothesis(models_b_repeats, testloader_b, device, 
                                       save_dir=os.path.join(analysis_dir, 'generalization'), use_cache=use_cache)
    
    # Run the plasticity analysis.
    if args.all or args.plasticity:
        print("Running plasticity analysis...")
        test_plasticity_hypothesis(models_b_repeats, testloader_a, testloader_b, device,
                                   save_dir=os.path.join(analysis_dir, 'plasticity'), use_cache=use_cache)
        
    # Run the adversarial analysis.
    if args.all or args.adversarial:
        print("Running adversarial analysis...")

        test_adversarial_hypothesis(models_b_repeats, testloader_b, device, 
                                    save_dir=os.path.join(analysis_dir, 'adversarial'), epsilon=0.05)

        test_adversarial_hypothesis(models_b_repeats, testloader_b, device, 
                                    save_dir=os.path.join(analysis_dir, 'adversarial'), epsilon=0.3)

if __name__ == '__main__':
    main()
