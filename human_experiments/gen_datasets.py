"""Makes the human experiment dataset."""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import os
import shutil

from parse_config import parse_config
from analysis import load_models

from torchattacks import PGD
import torchvision


def make_imgs_conv(model, dataset, n_imgs: int = 100):
    """Uses the convolution of the first layer to make the dataset."""

    imgs = []
    labels = []
    og_imgs = []
    for i in range(n_imgs):

        # Randomly select an image from the dataset.
        og_img, label = dataset[np.random.randint(len(dataset))]

        # Randomly sample a filter from the model.
        filter_idx = np.random.randint(model.g1.weight.shape[0])
        filter = model.g1.weight[filter_idx].detach()

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

        axs[i // 5, i % 5].imshow(og_img.permute(1, 2, 0), cmap='gray')
        axs[i // 5, i % 5].set_title(label)
        axs[i // 5, i % 5].axis('off')
        axs[i // 5, i % 5 + 5].imshow(img.permute(1, 2, 0), cmap='gray')
        axs[i // 5, i % 5 + 5].axis('off')

    plt.tight_layout()
    plt.show()


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to the configuration file that was run to train the models.")

    args = parser.parse_args()

    # Parse the configuration file + run the experiment.
    config = parse_config(args.config)
    config['save_dir'] = os.path.join(config['save_dir'], '0')

    # Load the models.
    models_a, models_b = load_models(config, intermediate=False)
    gabor_model_a = next(iter(models_a['gabor']['checkpoints'].values()))
    cnn_model_a = next(iter(models_a['cnn']['checkpoints'].values()))
    gabor_model_b = next(iter(models_b['gabor']['checkpoints'].values()))
    cnn_model_b = next(iter(models_b['cnn']['checkpoints'].values()))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the datasets.
    dataset_a_train, dataset_a_test = config['datasets']['initial']
    dataset_b_train, dataset_b_test = config['datasets']['finetune']

    n_exs = 25
    dataloader_a = torch.utils.data.DataLoader(dataset_a_test, batch_size=n_exs, shuffle=True)
    dataloader_b = torch.utils.data.DataLoader(dataset_b_test, batch_size=n_exs, shuffle=True)

    # # Make the images by convoluting the first layer of the model.
    # gabor_imgs_a, gabor_labels_a, og_imgs_a = make_imgs(gabor_model, dataset_a_train)
    # gabor_imgs_b, gabor_labels_b, og_imgs_b = make_imgs(gabor_model, dataset_b_train)
    # cnn_imgs_a, cnn_labels_a, og_imgs_a = make_imgs(cnn_model, dataset_a_train)
    # cnn_imgs_b, cnn_labels_b, og_imgs_b = make_imgs(cnn_model, dataset_b_train)

    # # Plot the images.
    # plot_imgs(gabor_imgs_a, gabor_labels_a, og_imgs_a)
    # plot_imgs(gabor_imgs_b, gabor_labels_b, og_imgs_b)
    # plot_imgs(cnn_imgs_a, cnn_labels_a, og_imgs_a)
    # plot_imgs(cnn_imgs_b, cnn_labels_b, og_imgs_b)

    # Remove old dataset.
    if os.path.isdir("human_experiments/data/"):
        shutil.rmtree("human_experiments/data/")

    # Construct the adversarial examples dataset.
    eps = 0.2
    for gabor_model, cnn_model, dataloader, name in (
        (gabor_model_a, cnn_model_a, dataloader_a, 'a'), 
        (gabor_model_b, cnn_model_b, dataloader_b, 'b')):

        # Make the adversarial examples.
        _, n_channels, _, _ = next(iter(dataloader))[0].shape
        cnn_atk = PGD(cnn_model, eps=eps, alpha=2/225, steps=10, random_start=True)
        gabor_atk = PGD(gabor_model, eps=eps, alpha=2/225, steps=10, random_start=True)
        cnn_atk.set_normalization_used(mean=(0.5,) * n_channels, std=(0.5,) * n_channels)
        gabor_atk.set_normalization_used(mean=(0.5,) * n_channels, std=(0.5,) * n_channels)

        data, labels = next(iter(dataloader))
        data, labels = data.to(device), labels.to(device)
        
        cnn_adv = cnn_atk(data, labels)
        gabor_adv = gabor_atk(data, labels)

        # # Plot the adversarial examples.
        # plot_imgs(cnn_adv, labels, data)
        # plot_imgs(gabor_adv, labels, data)

        # Save the adversarial examples.
        adv_img_dir = f"human_experiments/data/adversarial_examples/{name}"
        os.makedirs(adv_img_dir, exist_ok=True)
        os.makedirs(os.path.join(adv_img_dir, "gabor"), exist_ok=True)
        os.makedirs(os.path.join(adv_img_dir, "cnn"), exist_ok=True)
        og_img_dir = f"human_experiments/data/og_img/{name}"
        os.makedirs(og_img_dir, exist_ok=True)
        for i, (cnn_adv_img, gabor_adv_img, og_img, label) in enumerate(zip(cnn_adv, gabor_adv, data, labels)):
            cnn_adv_img = (cnn_adv_img + 1) / 2
            gabor_adv_img = (gabor_adv_img + 1) / 2
            og_img = (og_img + 1) / 2

            cnn_adv_img = torchvision.transforms.ToPILImage()(cnn_adv_img.cpu())
            gabor_adv_img = torchvision.transforms.ToPILImage()(gabor_adv_img.cpu())
            og_img = torchvision.transforms.ToPILImage()(og_img.cpu())

            cnn_adv_img.save(os.path.join(adv_img_dir, "cnn", f"{i}_{label}.png"))
            gabor_adv_img.save(os.path.join(adv_img_dir, "gabor", f"{i}_{label}.png"))
            og_img.save(os.path.join(og_img_dir, f"{i}_{label}.png"))


if __name__ == '__main__':
    main()
