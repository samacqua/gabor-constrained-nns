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

import torch
from torchvision import utils
import torch.nn.functional as F

from parse_config import parse_config

import warnings
warnings.filterwarnings("ignore")


def test_generalization_hypothesis(gabor_constrained_models: dict[int, list[torch.nn.Module]], 
                                   unconstrained_models: dict[int, list[torch.nn.Module]], 
                                   baseline_models: dict[int, list[torch.nn.Module]], test_loader_b, device, 
                                   save_dir: str):
    """Tests the hypothesis: Gabor-constrained neural networks will finetune better (converge faster + higher acc).
    
    Args:
        gabor_constrained_models: A dictionary of list of Gabor model checkpoints, indexed by epoch.
        unconstrained_models: A dictionary of list of CNN model checkpoints, indexed by epoch.
        test_loader_b: The dataloader for the test set of dataset B.
        device: The device to run the models on.
    """
    epochs = sorted(list(gabor_constrained_models.keys()))

    # Get the accuracy of each model at each stage of convergence.
    try:
        with open(os.path.join(save_dir, "gabor_accuracies.pkl"), "rb") as f:
            gabor_accuracies = pickle.load(f)
        with open(os.path.join(save_dir, "unconstrained_accuracies.pkl"), "rb") as f:
            unconstrained_accuracies = pickle.load(f)
        with open(os.path.join(save_dir, "baseline_accuracies.pkl"), "rb") as f:
            baseline_accuracies = pickle.load(f)
    except FileNotFoundError:
        gabor_accuracies = {}
        unconstrained_accuracies = {}
        baseline_accuracies = {}
        for epoch in tqdm(epochs):

            # Get the accuracy of each model.
            for gabor_model, unconstrained_model, baseline_model in zip(
                gabor_constrained_models[epoch], unconstrained_models[epoch], baseline_models[epoch]):

                gabor_acc = test_accuracy(test_loader_b, gabor_model, device)
                unconstrained_acc = test_accuracy(test_loader_b, unconstrained_model, device)
                baseline_acc = test_accuracy(test_loader_b, baseline_model, device)

                gabor_accuracies.setdefault(epoch, []).append(gabor_acc)
                unconstrained_accuracies.setdefault(epoch, []).append(unconstrained_acc)
                baseline_accuracies.setdefault(epoch, []).append(baseline_acc)

            # TODO: Check if the difference is statistically significant.
            pass

        # Save the results.
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "gabor_accuracies.pkl"), "wb") as f:
            pickle.dump(gabor_accuracies, f)
        with open(os.path.join(save_dir, "unconstrained_accuracies.pkl"), "wb") as f:
            pickle.dump(unconstrained_accuracies, f)
        with open(os.path.join(save_dir, "baseline_accuracies.pkl"), "wb") as f:
            pickle.dump(baseline_accuracies, f)

    # Calculate the mean accuracy of each model at each stage of convergence.
    gabor_mean_accuracies = {epoch: np.mean(accs) for epoch, accs in gabor_accuracies.items()}
    unconstrained_mean_accuracies = {epoch: np.mean(accs) for epoch, accs in unconstrained_accuracies.items()}
    baseline_mean_accuracies = {epoch: np.mean(accs) for epoch, accs in baseline_accuracies.items()}

    # Plot the accuracy of each model at each stage of convergence.
    fig, ax = plt.subplots()
    ax.plot(gabor_mean_accuracies.keys(), gabor_mean_accuracies.values(), label="Gabor")
    ax.plot(unconstrained_mean_accuracies.keys(), unconstrained_mean_accuracies.values(), label="Unconstrained")
    ax.plot(baseline_mean_accuracies.keys(), baseline_mean_accuracies.values(), label="Baseline")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy of Gabor and Unconstrained Models at Each Stage of Convergence")
    ax.legend()

    plt.savefig(os.path.join(save_dir, "accuracy_vs_epoch.png"))
    plt.show()


def test_plasticity_hypothesis(gabor_constrained_models, unconstrained_models, baseline_models, test_loader_a, 
                               test_loader_b, device, save_dir):
    """Tests the hypothesis: Gabor-constrained neural networks will retain original performance better.

    Expects that the models are finetuned to the same accuracy or loss.
    
    Args:
        gabor_constrained_models: A dictionary of list of Gabor model checkpoints, indexed by epoch.
        unconstrained_models: A dictionary of list of CNN model checkpoints, indexed by epoch.
        test_loader_a: The dataloader for the test set of dataset A.
        test_loader_b: The dataloader for the test set of dataset B.
        device: The device to run the models on.
    """

    # TODO: Check that actually tuned to same loss.

    epochs = sorted(list(gabor_constrained_models.keys()))

    # Check if the accuracy on dataset A, after finetuning to accuracy X on dataset B, is higher for the constrained
    # model than the unconstrained model.
    try:
        with open(os.path.join(save_dir, "gabor_accuracies.pkl"), "rb") as f:
            gabor_accuracies = pickle.load(f)
        with open(os.path.join(save_dir, "unconstrained_accuracies.pkl"), "rb") as f:
            unconstrained_accuracies = pickle.load(f)
        with open(os.path.join(save_dir, "baseline_accuracies.pkl"), "rb") as f:
            baseline_accuracies = pickle.load(f)
    except FileNotFoundError:
        gabor_accuracies = {}
        unconstrained_accuracies = {}
        baseline_accuracies = {}
        for epoch in tqdm(epochs):

            # Get the accuracy of each model.
            for gabor_model, unconstrained_model, baseline_model in zip(
                gabor_constrained_models[epoch], unconstrained_models[epoch], baseline_models[epoch]):

                gabor_acc_a = test_accuracy(test_loader_a, gabor_model, device)
                unconstrained_acc_a = test_accuracy(test_loader_a, unconstrained_model, device)
                baseline_acc_a = test_accuracy(test_loader_a, baseline_model, device)

                gabor_acc_b = test_accuracy(test_loader_b, gabor_model, device)
                unconstrained_acc_b = test_accuracy(test_loader_b, unconstrained_model, device)
                baseline_acc_b = test_accuracy(test_loader_b, baseline_model, device)

                gabor_accuracies.setdefault(gabor_acc_b, []).append(gabor_acc_a)
                unconstrained_accuracies.setdefault(unconstrained_acc_b, []).append(unconstrained_acc_a)
                baseline_accuracies.setdefault(baseline_acc_b, []).append(baseline_acc_a)

            # TODO: Check if the difference is statistically significant.
            pass

        # Save the results.
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "gabor_accuracies.pkl"), "wb") as f:
            pickle.dump(gabor_accuracies, f)
        with open(os.path.join(save_dir, "unconstrained_accuracies.pkl"), "wb") as f:
            pickle.dump(unconstrained_accuracies, f)
        with open(os.path.join(save_dir, "baseline_accuracies.pkl"), "wb") as f:
            pickle.dump(baseline_accuracies, f)

    # Calculate the mean accuracy of each model on dataset A at each stage of convergence of training on B.
    gabor_mean_accuracies = {acc_b: np.mean(accs_a) for acc_b, accs_a in gabor_accuracies.items()}
    unconstrained_mean_accuracies = {acc_b: np.mean(accs_a) for acc_b, accs_a in unconstrained_accuracies.items()}
    baseline_mean_accuracies = {acc_b: np.mean(accs_a) for acc_b, accs_a in baseline_accuracies.items()}

    # Plot the accuracy of each model at each stage of convergence.
    fig, ax = plt.subplots()
    ax.plot(gabor_mean_accuracies.keys(), gabor_mean_accuracies.values(), '-o', label="Gabor")
    ax.plot(unconstrained_mean_accuracies.keys(), unconstrained_mean_accuracies.values(), '-o', label="Unconstrained")
    ax.plot(baseline_mean_accuracies.keys(), baseline_mean_accuracies.values(), '-o', label="Baseline")
    ax.set_xlabel("Accuracy on Dataset B")
    ax.set_ylabel("Accuracy on Dataset A")
    ax.set_title("Accuracy of Gabor and Unconstrained Models at Each Stage of Convergence")
    ax.legend()

    plt.savefig(os.path.join(save_dir, "accuracy_vs_accuracy.png"))
    plt.show()


def fgsm_attack(image, epsilon, data_grad):
    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon*sign_data_grad
    # Adding clipping to maintain [-1,1] range
    perturbed_image = torch.clamp(perturbed_image, -1, 1)
    # Return the perturbed image
    return perturbed_image

  
def adversarial_attack(model, x, y, device, epsilon=0.1):
    """Returns the adversarial example that is perturbed by epsilon."""

    x = x.clone().unsqueeze(0)
    x.requires_grad = True

    # Forward pass the data through the model + backpropogate the error.
    output = model(x)
    loss = torch.nn.CrossEntropyLoss()(output, y.unsqueeze(0))
    model.zero_grad()
    loss.backward()
    data_grad = x.grad.data

    # Call FGSM Attack
    x_adv = fgsm_attack(x, epsilon, data_grad)

    # Re-classify the perturbed image
    output_adversarial = model(x_adv)

    return x_adv, output, output_adversarial


def test_adversarial_hypothesis(gabor_constrained_model, unconstrained_model, test_loader, device, save_dir, epsilon=0.1):
    """Tests the hypothesis: Gabor-constrained neural networks will be more robust to adversarial attacks.

    Expects that the models are finetuned to the same accuracy or loss.
    
    Args:
        gabor_constrained_model: The Gabor model.
        unconstrained_model: The CNN model.
        test_loader: The dataloader for the test set.
        device: The device to run the models on.
    """

    # For each model, sample examples that it gets correct.
    gabor_correct = []
    unconstrained_correct = []
    N_exs = 500
    for model, list_of_correct in zip([gabor_constrained_model, unconstrained_model], [gabor_correct, unconstrained_correct]):
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            y_pred = model(x)
            y_pred = torch.argmax(y_pred, dim=1)
            for i in range(len(x)):
                if y_pred[i] == y[i]:
                    list_of_correct.append((x[i], y[i]))
                    if len(list_of_correct) == N_exs:
                        break
            if len(list_of_correct) == N_exs:
                break


    if len(gabor_correct) < N_exs or len(unconstrained_correct) < N_exs:
        raise ValueError("Not enough examples were sampled.")

    # For each high-confidence instance, find the adversarial example that makes it classified as the wrong class.
    gabor_adversaries = []
    unconstrained_adversaries = []
    for model, correct_data, adversaries in zip(
        [gabor_constrained_model, unconstrained_model], 
        [gabor_correct, unconstrained_correct], 
        [gabor_adversaries, unconstrained_adversaries]):

        for x, y in tqdm(correct_data):
            x_adv, og_output, adversarial_output = adversarial_attack(model, x, y, device, epsilon=epsilon)
            
            # Normalize the outputs.
            og_output = torch.nn.functional.softmax(og_output, dim=1)[0,y.item()]
            adversarial_output = torch.nn.functional.softmax(adversarial_output, dim=1)[0,y.item()]

            adversaries.append((x, x_adv, y, og_output, adversarial_output))

    # Calculate the difference in confidence between the original and adversarial examples.
    gabor_distances = []
    unconstrained_distances = []
    gabor_img_diffs = []
    unconstrained_img_diffs = []

    for gabor_adv, unconstrained_adv in zip(gabor_adversaries, unconstrained_adversaries):

        x, x_adv, y, og_output, adversarial_output = gabor_adv
        gabor_distances.append(torch.abs(og_output - adversarial_output).item())
        gabor_img_diffs.append(torch.norm(x - x_adv).item())

        x, x_adv, y, og_output, adversarial_output = unconstrained_adv
        unconstrained_distances.append(torch.abs(og_output - adversarial_output).item())
        unconstrained_img_diffs.append(torch.norm(x - x_adv).item())

    # Check if statistically significant.
    # Between the distances.
    t, p = stats.ttest_ind(gabor_distances, unconstrained_distances)
    print(f"conf t: {t}, p: {p}")

    # Between the image differences.
    t, p = stats.ttest_ind(gabor_img_diffs, unconstrained_img_diffs)
    print(f"img t: {t}, p: {p}")

    # Print the result.
    print(f"Gabor confidence difference: {np.mean(gabor_distances):.3f} {np.std(gabor_distances):.3f}")
    print(f"Unconstrained confidence difference: {np.mean(unconstrained_distances):.3f} {np.std(unconstrained_distances):.3f}")
    print(f"Gabor image difference: {np.mean(gabor_img_diffs):.3f} {np.std(gabor_img_diffs):.3f}")
    print(f"Unconstrained image difference: {np.mean(unconstrained_img_diffs):.3f} {np.std(unconstrained_img_diffs):.3f}")


def visualize_tensor(tensor: torch.Tensor, ch: int = 0, allkernels: bool = False, nrow: int = 8, padding: int = 1, save_dir: str = None): 
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

    if save_dir is not None:
        plt.savefig(os.path.join(save_dir, "features.png"))

    plt.show()


def visualize_features(model: torch.nn.Module, save_dir: str):
    """Visualizes the features of a model."""

    # Get the weights of the first layer.
    weights = model.conv1.weight.data.cpu()
    visualize_tensor(weights, ch=0, allkernels=False, save_dir=save_dir)


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
        model.load_state_dict(torch.load(save_path))
    except FileNotFoundError:
        print("Could not load model from path: {}".format(save_path))
        model = None

    return model


def load_models(config: dict, intermediate: bool = False) -> dict[str, tuple[torch.nn.Module, torch.nn.Module]]:
    """Loads the final models."""

    # TODO: Load across different runs.

    model_save_dir = os.path.join(config['save_dir'], "models")
    base_model = config['base_model']
    n_channels = config['datasets']['params']['n_channels']

    models = {}
    for model_sequence in os.listdir(model_save_dir):

        # If not simple gabor or cnn.
        if model_sequence not in ['gabor', 'cnn', 'baseline']:
            continue

        sequence_path = os.path.join(model_save_dir, model_sequence)

        # Load the model after training on the first dataset.
        model_a_path = os.path.join(sequence_path, "model_a.pt")
        init_cfg = config['schedules'][model_sequence]['initial_train']
        model_base = config['schedules'][model_sequence]['model']
        model_a = model_base    # load_model(model_base, init_cfg['gabor_constrained'], n_channels=n_channels, save_path=model_a_path)

        # Load the model after training on the second dataset.
        model_b_path = os.path.join(sequence_path, "model_b.pt")
        finetune_cfg = config['schedules'][model_sequence]['finetune']
        model_b = model_base    # load_model(model_base, finetune_cfg['gabor_constrained'], n_channels=n_channels, save_path=model_b_path)

        # Load models saved during training. Only load if specified to save loading time.
        intermediate_a = {}
        intermediate_b = {}
        if intermediate:
            for model_fname in os.listdir(os.path.join(sequence_path)):
                # Has form model_{a or b}.pt or model_{a or b}_{epoch}.pt
                _, a_or_b, *args = model_fname.split("_")

                if args:    # If there is an epoch number.
                    epoch = int(args[0].split(".")[0])

                    # Load the model.
                    intermediate_cfg = init_cfg if a_or_b == "a" else finetune_cfg
                    intermediate_model = load_model(base_model, intermediate_cfg['gabor_constrained'], 
                                                    n_channels=n_channels, 
                                                    save_path=os.path.join(sequence_path, model_fname))

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
    parser.add_argument("--adversarial", action="store_true", help="Run the adversarial analysis.")

    args = parser.parse_args()

    # Parse the configuration file + run the experiment.
    config = parse_config(args.config)
    config['save_dir'] = os.path.join(config['save_dir'], 'repeat_1')   # TODO: handle multiple runs.

    # Create the analysis folder.
    analysis_dir = os.path.join(config['save_dir'], "analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    # Load the models.
    models = load_models(config, intermediate=args.all or args.generalization or args.plasticity)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the datasets.
    if args.all or args.generalization or args.plasticity or args.test or args.adversarial:
        print("Loading datasets...")
        testloader_a = torch.utils.data.DataLoader(config['datasets']['initial'][1], batch_size=128, shuffle=False)
        testloader_b = torch.utils.data.DataLoader(config['datasets']['finetune'][1], batch_size=128, shuffle=False)

    # Test the final accuracies of the models.
    if args.all or args.test:
        print("Testing accuracies...")
        accuracies = {}
        for model_sequence, (model_a, model_b, _, _) in models.items():
            model_a_acc = test_accuracy(testloader_a, model_a, device) if model_a is not None else None
            accuracies[model_sequence] = (model_a_acc, 
                                        test_accuracy(testloader_b, model_b, device),
                                        test_accuracy(testloader_a, model_b, device))

        # Print the results.
        max_name_len = max([len(name) for name in accuracies])
        padded_name = "Name" + ' ' * (max_name_len - 4)
        print(f"{padded_name}\tA\tB\tA (trained on B)")
        for model_sequence, (acc_a, acc_b, acc_ba) in accuracies.items():
            padded_name = model_sequence + ' ' * (max_name_len - len(model_sequence))
            print(f"{padded_name}\t{acc_a}\t{acc_b}\t{acc_ba}")

        # Save the results.
        with open(os.path.join(analysis_dir, "test_accuracies.pkl"), "wb") as f:
            pickle.dump(accuracies, f)

    # Visualize the features of each model.
    if args.all or args.visualize:
        print("Visualizing features...")
        for model_sequence, (model_a, model_b, _, _) in models.items():
            # visualize_features(model_a, save_dir=analysis_dir)
            visualize_features(model_b, save_dir=analysis_dir)

    # Run the generalization analysis.
    if args.all or args.generalization:
        print("Running generalization analysis...")

        # Load the intermediate models on the finetuned dataset.
        gabor_finetune_model_checkpoints = models['gabor'][3]
        cnn_finetune_model_checkpoints = models['cnn'][3]
        baseline_model_checkpoints = models['baseline'][3]

        test_generalization_hypothesis(gabor_finetune_model_checkpoints, cnn_finetune_model_checkpoints, 
                                       baseline_model_checkpoints, testloader_b, device, 
                                       save_dir=os.path.join(analysis_dir, 'generalization'))

    
    # Run the plasticity analysis.
    if args.all or args.plasticity:
        print("Running plasticity analysis...")

        # Load the intermediate models on the finetuned dataset.
        gabor_finetune_model_checkpoints = models['gabor'][3]
        cnn_finetune_model_checkpoints = models['cnn'][3]
        baseline_model_checkpoints = models['baseline'][3]

        test_plasticity_hypothesis(gabor_finetune_model_checkpoints, cnn_finetune_model_checkpoints, 
                                   baseline_model_checkpoints, testloader_a, testloader_b, device,
                                   save_dir=os.path.join(analysis_dir, 'plasticity'))
        
    # Run the adversarial analysis.
    if args.all or args.adversarial:
        print("Running adversarial analysis...")

        gabor_finetune_model = models['gabor'][1]
        cnn_finetune_model = models['cnn'][1]
        test_adversarial_hypothesis(gabor_finetune_model, cnn_finetune_model, testloader_b, device, 
                                    save_dir=os.path.join(analysis_dir, 'adversarial'), epsilon=0.05)

        test_adversarial_hypothesis(gabor_finetune_model, cnn_finetune_model, testloader_b, device, 
                                    save_dir=os.path.join(analysis_dir, 'adversarial'), epsilon=0.3)

if __name__ == '__main__':
    main()
