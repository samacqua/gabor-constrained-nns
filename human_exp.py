"""Makes the human experiment dataset."""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import os

from parse_config import parse_config
from analysis import load_models


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


def main():
    
    parser = argparse.ArgumentParser(description="Run experiments.")
    parser.add_argument("config", type=str, help="Path to the configuration file.")

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
    dataset_a = config['initial_dataset'][0]
    dataset_b = config['finetune_dataset'][0]

    # Make the images.
    gabor_imgs_a, gabor_labels_a, og_imgs_a = make_imgs(gabor_model, dataset_a)
    gabor_imgs_b, gabor_labels_b, og_imgs_b = make_imgs(gabor_model, dataset_b)
    cnn_imgs_a, cnn_labels_a, og_imgs_a = make_imgs(cnn_model, dataset_a)
    cnn_imgs_b, cnn_labels_b, og_imgs_b = make_imgs(cnn_model, dataset_b)

    # Plot the images.
    plot_imgs(gabor_imgs_a, gabor_labels_a, og_imgs_a)
    plot_imgs(gabor_imgs_b, gabor_labels_b, og_imgs_b)
    plot_imgs(cnn_imgs_a, cnn_labels_a, og_imgs_a)
    plot_imgs(cnn_imgs_b, cnn_labels_b, og_imgs_b)

if __name__ == '__main__':
    main()
