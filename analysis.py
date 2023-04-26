"""Analysis of the results of the experiments + code to generate figures for final paper."""

import torch
import argparse
import os
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from torchvision import utils

from parse_config import parse_config


def test_generalization_hypothesis(gabor_constrained_models, unconstrained_models):
    """Tests the hypothesis: Gabor-constrained neural networks will finetune better (converge faster + higher acc)."""

    # Check if statistically higher accuracy at each stage of convergence.

    # Plot the accuracy of each model at each stage of convergence.

    raise NotImplementedError

def test_plasticity_hypothesis(gabor_constrained_models, unconstrained_models):
    """Tests the hypothesis: Gabor-constrained neural networks will retain original performance better."""

    # Expects that the models are finetuned to the same accuracy.

    # Check if the accuracy on dataset A, after finetuning to accuracy X on dataset B, is higher for the constrained
    # model than the unconstrained model.

    # Plot the accuracy of each model on dataset A at each stage of convergence on dataset B.

    raise NotImplementedError

def visTensor(tensor, ch=0, allkernels=False, nrow=8, padding=1): 
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
    visTensor(weights, ch=0, allkernels=False)



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


def load_models(config: dict) -> dict[str, tuple[torch.nn.Module, torch.nn.Module]]:
    """Loads the models."""

    model_save_dir = os.path.join(config['save_dir'], "models")
    base_model = config['base_model']

    models = {}
    for model_sequence in os.listdir(model_save_dir):

        # Load the model after training on the first dataset.
        model_a_path = os.path.join(model_save_dir, model_sequence, "model_a.pt")
        init_cfg = config['schedules'][model_sequence]['initial_train']
        model_a = base_model(is_gabornet=init_cfg['gabor_constrained'], n_channels=config['n_channels'])
        model_a.load_state_dict(torch.load(model_a_path))

        # Load the model after training on the second dataset.
        model_b_path = os.path.join(model_save_dir, model_sequence, "model_b.pt")
        finetune_cfg = config['schedules'][model_sequence]['finetune']
        model_b = base_model(finetune_cfg['gabor_constrained'], n_channels=config['n_channels'])
        model_b.load_state_dict(torch.load(model_b_path))

        models[model_sequence] = (model_a, model_b)

    return models


def main():

    # Parse command line arguments.
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")

    args = parser.parse_args()

    # Parse the configuration file + run the experiment.
    config = parse_config(args.config)

    # Load the models.
    models = load_models(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # # Test the accuracy of each model.
    # testloader_a = torch.utils.data.DataLoader(config['initial_dataset'][1], **config['dataloader_params'])
    # testloader_b = torch.utils.data.DataLoader(config['finetune_dataset'][1], **config['dataloader_params'])

    # accuracies = {}
    # for model_sequence, (model_a, model_b) in models.items():
    #     accuracies[model_sequence] = (test_accuracy(testloader_a, model_a, device), 
    #                                   test_accuracy(testloader_b, model_b, device),
    #                                   test_accuracy(testloader_a, model_b, device))
        
    # # Print the results.
    # print("Model Sequence\tInitial Dataset\tFinetune Dataset\tInitial Dataset (Finetuned Model)")
    # for model_sequence, (acc_a, acc_b, acc_ba) in accuracies.items():
    #     print(f"{model_sequence}\t{acc_a}\t{acc_b}\t{acc_ba}")

    # Visualize the features of each model.
    for model_sequence, (model_a, model_b) in models.items():
        visualize_features(model_a, device)
        visualize_features(model_b, device)


if __name__ == '__main__':
    main()
