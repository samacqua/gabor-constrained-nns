"""Makes the human experiment dataset."""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import os

from parse_config import parse_config
from analysis import load_models, adversarial_attack


def make_imgs(model, dataset, n_imgs: int = 100):
    """Uses the convolution of the first layer to make the dataset."""

    imgs = []
    labels = []
    og_imgs = []
    for i in range(n_imgs):

        # Randomly select an image from the dataset.
        og_img, label = dataset[np.random.randint(len(dataset))]

        # Randomly sample a filter from the model.
        filter_idx = np.random.randint(model.conv1.weight.shape[0])
        filter = model.conv1.weight[filter_idx].detach()

        # Convolve the image with the filter.
        img = torch.nn.functional.conv2d(og_img.unsqueeze(0), filter.unsqueeze(1)).squeeze()

        imgs.append(img)
        labels.append(label)
        og_imgs.append(og_img.squeeze())

    return imgs, labels, og_imgs


def plot_imgs(imgs, labels, og_imgs):
    """Plots the images."""

    imgs = imgs[:25]
    labels = labels[:25]
    og_imgs = og_imgs[:25]

    # Plot each image next to its original.
    fig, axs = plt.subplots(5, 10, figsize=(10, 5))
    for i, (img, label, og_img) in enumerate(zip(imgs, labels, og_imgs)):

        axs[i // 5, i % 5].imshow(og_img, cmap='gray')
        axs[i // 5, i % 5].set_title(label)
        axs[i // 5, i % 5].axis('off')
        axs[i // 5, i % 5 + 5].imshow(img, cmap='gray')
        axs[i // 5, i % 5 + 5].axis('off')

    plt.tight_layout()
    plt.show()


def get_high_conf_answers(model_a, model_b, dataloader, device, min_conf: float = 0.8, n_exs: int = 500):
    """Gets examples from the dataset that both models get right confidently."""

    # For each model, sample examples that it gets correct.
    correct = []

    for x, y in dataloader:
        x, y = x.to(device), y.to(device)

        a_out = model_a(x)
        b_out = model_b(x)

        y_pred_a = torch.argmax(a_out, dim=1)
        y_pred_b = torch.argmax(b_out, dim=1)

        both_correct = torch.where(torch.logical_and((y_pred_a == y), (y_pred_b == y)))[0]

        for i in both_correct:
            if torch.nn.functional.sigmoid(a_out[i]).max() > min_conf and torch.nn.functional.sigmoid(b_out[i]).max() > min_conf:
                correct.append((x[i], y[i]))
                if len(correct) == n_exs:
                    break
        
        if len(correct) == n_exs:
            break

    return correct


def make_adversarial_examples(model, dataset, eps: float = 0.1):
    """Makes adversarial examples."""

    adversarial_exs = []
    conf_diffs = []
    for x, y in dataset:
        x_adv, og_out, new_out = adversarial_attack(model, x, y, device='cpu', epsilon=eps)
        adversarial_exs.append((x.squeeze().detach(), x_adv.squeeze().detach(), y.detach().item()))

        # Get the confidence difference.
        assert og_out.argmax() == y
        conf_diff = torch.nn.functional.sigmoid(new_out)[0,y] - torch.nn.functional.sigmoid(og_out)[0,y]
        print(conf_diff)
        conf_diffs.append(conf_diff.detach().item())

    return adversarial_exs, conf_diffs


def main():
    
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("config", type=str, help="Path to the configuration file that was run to train the models.")

    args = parser.parse_args()

    # Parse the configuration file + run the experiment.
    config = parse_config(args.config)
    config['save_dir'] = os.path.join(config['save_dir'], 'repeat_1')   # TODO: handle multiple runs.

    # Load the models.
    models = load_models(config, intermediate=False)
    gabor_model = models['gabor'][0]
    cnn_model = models['cnn'][0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the datasets.
    dataset_a_train, dataset_a_test = config['datasets']['initial']
    dataset_b_train, dataset_b_test = config['datasets']['finetune']

    # Make the images.
    gabor_imgs_a, gabor_labels_a, og_imgs_a = make_imgs(gabor_model, dataset_a_train)
    gabor_imgs_b, gabor_labels_b, og_imgs_b = make_imgs(gabor_model, dataset_b_train)
    cnn_imgs_a, cnn_labels_a, og_imgs_a = make_imgs(cnn_model, dataset_a_train)
    cnn_imgs_b, cnn_labels_b, og_imgs_b = make_imgs(cnn_model, dataset_b_train)

    # Plot the images.
    plot_imgs(gabor_imgs_a, gabor_labels_a, og_imgs_a)
    plot_imgs(gabor_imgs_b, gabor_labels_b, og_imgs_b)
    plot_imgs(cnn_imgs_a, cnn_labels_a, og_imgs_a)
    plot_imgs(cnn_imgs_b, cnn_labels_b, og_imgs_b)

    # # Get the high confidence examples.
    # dataloader_a = torch.utils.data.DataLoader(dataset_a_test, batch_size=32, shuffle=True)
    # dataloader_b = torch.utils.data.DataLoader(dataset_b_test, batch_size=32, shuffle=True)

    # high_conf_a = get_high_conf_answers(gabor_model, cnn_model, dataloader_a, device, n_exs=25)
    # high_conf_b = get_high_conf_answers(gabor_model, cnn_model, dataloader_b, device, n_exs=50)

    # # Make the adversarial examples.
    # # gabor_adv_a, _ = make_adversarial_examples(gabor_model, high_conf_a, eps=0.3)
    # # unconstrained_adv_a, _ = make_adversarial_examples(cnn_model, high_conf_a)
    # # make_adversarial_examples(gabor_model, high_conf_b)
    # # make_adversarial_examples(cnn_model, high_conf_b)

    # # Plot the adversarial examples.
    # plot_imgs([ex[0] for ex in gabor_adv_a], [ex[2] for ex in gabor_adv_a], [ex[1] for ex in gabor_adv_a])
    # plot_imgs([ex[0] for ex in unconstrained_adv_a], [ex[2] for ex in unconstrained_adv_a], [ex[1] for ex in unconstrained_adv_a])


if __name__ == '__main__':
    main()
