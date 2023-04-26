"""Analysis of the results of the experiments + code to generate figures for final paper."""

import argparse
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch
from torchvision import utils

from parse_config import parse_config

import warnings
warnings.filterwarnings("ignore")


def test_generalization_hypothesis(gabor_constrained_models: dict[int, list[torch.nn.Module]], 
                                   unconstrained_models: dict[int, list[torch.nn.Module]], test_loader_b, device):
    """Tests the hypothesis: Gabor-constrained neural networks will finetune better (converge faster + higher acc).
    
    Args:
        gabor_constrained_models: A dictionary of list of Gabor model checkpoints, indexed by epoch.
        unconstrained_models: A dictionary of list of CNN model checkpoints, indexed by epoch.
        test_loader_b: The dataloader for the test set of dataset B.
        device: The device to run the models on.
    """
    epochs = sorted(list(gabor_constrained_models.keys()))

    # Check if statistically higher accuracy at each stage of convergence.
    gabor_accuracies = {}
    unconstrained_accuracies = {}
    for epoch in tqdm(epochs):

        # Get the accuracy of each model.
        for gabor_model, unconstrained_model in zip(gabor_constrained_models[epoch], unconstrained_models[epoch]):
            gabor_acc = test_accuracy(test_loader_b, gabor_model, device)
            unconstrained_acc = test_accuracy(test_loader_b, unconstrained_model, device)

            gabor_accuracies.setdefault(epoch, []).append(gabor_acc)
            unconstrained_accuracies.setdefault(epoch, []).append(unconstrained_acc)

        # TODO: Check if the difference is statistically significant.
        pass

    # Calculate the mean accuracy of each model at each stage of convergence.
    gabor_mean_accuracies = {epoch: np.mean(accs) for epoch, accs in gabor_accuracies.items()}
    unconstrained_mean_accuracies = {epoch: np.mean(accs) for epoch, accs in unconstrained_accuracies.items()}

    # Plot the accuracy of each model at each stage of convergence.
    fig, ax = plt.subplots()
    ax.plot(gabor_mean_accuracies.keys(), gabor_mean_accuracies.values(), label="Gabor")
    ax.plot(unconstrained_mean_accuracies.keys(), unconstrained_mean_accuracies.values(), label="Unconstrained")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy of Gabor and Unconstrained Models at Each Stage of Convergence")
    ax.legend()
    plt.show()


def test_plasticity_hypothesis(gabor_constrained_models, unconstrained_models, test_loader_a, device):
    """Tests the hypothesis: Gabor-constrained neural networks will retain original performance better.

    Expects that the models are finetuned to the same accuracy or loss.
    
    Args:
        gabor_constrained_models: A dictionary of list of Gabor model checkpoints, indexed by epoch.
        unconstrained_models: A dictionary of list of CNN model checkpoints, indexed by epoch.
        test_loader_a: The dataloader for the test set of dataset A.
        device: The device to run the models on.
    """

    # TODO: Check that actually tuned to same loss.

    epochs = sorted(list(gabor_constrained_models.keys()))

    # Check if the accuracy on dataset A, after finetuning to accuracy X on dataset B, is higher for the constrained
    # model than the unconstrained model.
    gabor_accuracies = {}
    unconstrained_accuracies = {}
    for epoch in tqdm(epochs):

        # Get the accuracy of each model.
        for gabor_model, unconstrained_model in zip(gabor_constrained_models[epoch], unconstrained_models[epoch]):
            gabor_acc = test_accuracy(test_loader_a, gabor_model, device)
            unconstrained_acc = test_accuracy(test_loader_a, unconstrained_model, device)

            gabor_accuracies.setdefault(epoch, []).append(gabor_acc)
            unconstrained_accuracies.setdefault(epoch, []).append(unconstrained_acc)

        # TODO: Check if the difference is statistically significant.
        pass

    # Calculate the mean accuracy of each model on dataset A at each stage of convergence of training on B.
    gabor_mean_accuracies = {epoch: np.mean(accs) for epoch, accs in gabor_accuracies.items()}
    unconstrained_mean_accuracies = {epoch: np.mean(accs) for epoch, accs in unconstrained_accuracies.items()}

    # Plot the accuracy of each model at each stage of convergence.
    fig, ax = plt.subplots()
    ax.plot(gabor_mean_accuracies.keys(), gabor_mean_accuracies.values(), label="Gabor")
    ax.plot(unconstrained_mean_accuracies.keys(), unconstrained_mean_accuracies.values(), label="Unconstrained")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy of Gabor and Unconstrained Models at Each Stage of Convergence")
    ax.legend()
    plt.show()


def visualize_tensor(tensor: torch.Tensor, ch: int = 0, allkernels: bool = False, nrow: int = 8, padding: int = 1): 
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
    plt.show()


def visualize_features(model: torch.nn.Module, device: torch.device):
    """Visualizes the features of a model."""

    # Get the weights of the first layer.
    weights = model.conv1.weight.data.cpu()
    visualize_tensor(weights, ch=0, allkernels=False)


def test_accuracy(test_loader: torch.utils.data.DataLoader, model: torch.nn.Module, device: torch.device):
    """Tests the accuracy of a model."""
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    return correct / total


def load_models(config: dict, intermediate: bool = False) -> dict[str, tuple[torch.nn.Module, torch.nn.Module]]:
    """Loads the final models."""

    model_save_dir = os.path.join(config['save_dir'], "models")
    base_model = config['base_model']

    models = {}
    for model_sequence in os.listdir(model_save_dir):

        sequence_path = os.path.join(model_save_dir, model_sequence)

        # Load the model after training on the first dataset.
        model_a_path = os.path.join(sequence_path, "model_a.pt")
        init_cfg = config['schedules'][model_sequence]['initial_train']
        model_a = base_model(is_gabornet=init_cfg['gabor_constrained'], n_channels=config['n_channels'])
        model_a.load_state_dict(torch.load(model_a_path))

        # Load the model after training on the second dataset.
        model_b_path = os.path.join(sequence_path, "model_b.pt")
        finetune_cfg = config['schedules'][model_sequence]['finetune']
        model_b = base_model(finetune_cfg['gabor_constrained'], n_channels=config['n_channels'])
        model_b.load_state_dict(torch.load(model_b_path))

        # Load models saved during training. Only load if specified to save loading time.
        intermediate_a = {}
        intermediate_b = {}
        if intermediate:
            for model_fname in os.listdir(os.path.join(sequence_path)):
                # Has form model_{a or b}.pt or model_{a or b}_{epoch}.pt
                _, a_or_b, *args = model_fname.split("_")

                if args:    # If there is an epoch number.
                    epoch = int(args[0].split(".")[0])

                    intermediate_model = deepcopy(model_a if a_or_b == "a" else model_b)
                    intermediate_model.load_state_dict(torch.load(os.path.join(sequence_path, model_fname)))

                    if a_or_b == "a":
                        intermediate_a[epoch] = [intermediate_model]
                    else:
                        intermediate_b[epoch] = [intermediate_model]

        models[model_sequence] = (model_a, model_b, intermediate_a, intermediate_b)

    return models


def main():

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")
    parser.add_argument("--all", action="store_true", help="Run all the analyses.")

    parser.add_argument("--test", action="store_true", help="Calculate test accuracy of the models.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the features of the models.")
    parser.add_argument("--generalization", action="store_true", help="Run the generalization analysis.")
    parser.add_argument("--plasticity", action="store_true", help="Run the plasticity analysis.")

    args = parser.parse_args()

    # Parse the configuration file + run the experiment.
    config = parse_config(args.config)

    # Load the models.
    models = load_models(config, intermediate=args.all or args.generalization or args.plasticity)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the datasets.
    if args.all or args.generalization or args.plasticity or args.test:
        print("Loading datasets...")
        testloader_a = torch.utils.data.DataLoader(config['initial_dataset'][1], **config['dataloader_params'])
        testloader_b = torch.utils.data.DataLoader(config['finetune_dataset'][1], **config['dataloader_params'])

    # Test the final accuracies of the models.
    if args.all or args.test:
        print("Testing accuracies...")
        accuracies = {}
        for model_sequence, (model_a, model_b, _, _) in models.items():
            accuracies[model_sequence] = (test_accuracy(testloader_a, model_a, device), 
                                        test_accuracy(testloader_b, model_b, device),
                                        test_accuracy(testloader_a, model_b, device))
            
        # Print the results.
        print("Model Sequence\tInitial Dataset\tFinetune Dataset\tInitial Dataset (Finetuned Model)")
        for model_sequence, (acc_a, acc_b, acc_ba) in accuracies.items():
            print(f"{model_sequence}\t{acc_a}\t{acc_b}\t{acc_ba}")

    # Visualize the features of each model.
    if args.all or args.visualize:
        print("Visualizing features...")
        for model_sequence, (model_a, model_b, _, _) in models.items():
            visualize_features(model_a, device)
            # visualize_features(model_b, device)

    # Run the generalization analysis.
    if args.all or args.generalization:
        print("Running generalization analysis...")

        # Load the intermediate models on the finetuned dataset.
        gabor_finetune_model_checkpoints = models['gabor'][3]
        cnn_finetune_model_checkpoints = models['cnn'][3]

        test_generalization_hypothesis(gabor_finetune_model_checkpoints, cnn_finetune_model_checkpoints, testloader_b, device)

    
    # Run the plasticity analysis.
    if args.all or args.plasticity:
        print("Running plasticity analysis...")

        # Load the intermediate models on the finetuned dataset.
        gabor_finetune_model_checkpoints = models['gabor'][3]
        cnn_finetune_model_checkpoints = models['cnn'][3]

        test_generalization_hypothesis(gabor_finetune_model_checkpoints, cnn_finetune_model_checkpoints, testloader_a, device)


if __name__ == '__main__':
    main()
