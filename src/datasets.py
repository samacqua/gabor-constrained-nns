import torchvision
import torchvision.transforms as transforms
import torch
import os

from recreate.dataset import DogsCatsDataset


def load_dataset(dataset_name: str, img_size: tuple[int, int], n_channels: int, dataset_dir: str = "./data"
                 ) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Loads a train + test dataset."""

    assert n_channels in (1, 3), f"n_channels must be 1 or 3, not {n_channels}"

    # Make the transforms based on the number of channels.
    # CIFAR-10 is 3-channel, Fashion-MNIST is 1-channel, so 1 has to change.
    # If 3 channels, then just copying Fashion-MNIST across all channels.
    # If 1 channel, then converting CIFAR-10 to grayscale.
    cifar_transforms = ([torchvision.transforms.Grayscale(num_output_channels=1)] if n_channels == 1 else []) + [
        torchvision.transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * n_channels, (0.5,) * n_channels)
    ]
    fashion_mnist_transforms = ([torchvision.transforms.Grayscale(num_output_channels=3)] if n_channels == 3 else []) + [
        torchvision.transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * n_channels, (0.5,) * n_channels)
    ]
    cats_dogs_transforms = ([torchvision.transforms.Grayscale(num_output_channels=1)] if n_channels == 1 else []) + [
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * n_channels, (0.5,) * n_channels),
    ]

    # Maps the dataset name to the loader of that dataset.
    datasets = {
        "cifar10": lambda train: torchvision.datasets.CIFAR10(
            root=dataset_dir, train=train, download=True, transform=transforms.Compose(cifar_transforms)),
        "fashion-mnist": lambda train: torchvision.datasets.FashionMNIST(
            root=dataset_dir, train=train, download=True, transform=transforms.Compose(fashion_mnist_transforms)),
        "dogs-vs-cats": lambda train: DogsCatsDataset(
            root_dir=os.path.join(dataset_dir, "train" if train else "test1"), 
            transform=transforms.Compose(cats_dogs_transforms))
    }

    # Load the dataset.
    if dataset_name not in datasets:
        raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets: {list(datasets.keys())}")

    dataset = datasets[dataset_name]
    trainset = dataset(train=True)
    testset = dataset(train=False)
    
    return trainset, testset
