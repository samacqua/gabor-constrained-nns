"""Analysis of the results of the experiments + code to generate figures for final paper."""

import torch
import argparse
import os
from copy import deepcopy

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


def visualize_features(model: torch.nn.Module, device: torch.device):
    """Visualizes the features of a model."""
    raise NotImplementedError


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
        model_a = deepcopy(base_model)
        a_is_gabor = config['schedules'][model_sequence]['initial_train']['gabor_constrained']
        model_a.change_constraint(gabor_constrained=a_is_gabor)
        model_a.load_state_dict(torch.load(model_a_path))

        # Load the model after training on the second dataset.
        model_b_path = os.path.join(model_save_dir, model_sequence, "model_b.pt")
        model_b = deepcopy(base_model)
        b_is_gabor = config['schedules'][model_sequence]['finetune']['gabor_constrained']
        model_b.change_constraint(gabor_constrained=b_is_gabor)
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

    # Test the accuracy of each model.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    testloader_a = torch.utils.data.DataLoader(config['initial_dataset'][1], **config['dataloader_params'])
    testloader_b = torch.utils.data.DataLoader(config['finetune_dataset'][1], **config['dataloader_params'])

    accuracies = {}
    for model_sequence, (model_a, model_b) in models.items():
        accuracies[model_sequence] = (test_accuracy(testloader_a, model_a, device), 
                                      test_accuracy(testloader_b, model_b, device),
                                      test_accuracy(testloader_a, model_b, device))
        
    # Print the results.
    print("Model Sequence\tInitial Dataset\tFinetune Dataset\tInitial Dataset (Finetuned Model)")
    for model_sequence, (acc_a, acc_b, acc_ba) in accuracies.items():
        print(f"{model_sequence}\t{acc_a}\t{acc_b}\t{acc_ba}")


if __name__ == '__main__':
    main()
