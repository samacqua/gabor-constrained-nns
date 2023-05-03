"""From https://github.com/iKintosh/GaborNet/blob/master/sanity_check/dataset.py."""

import os

from skimage import io
from torch.utils.data import Dataset


class DogsCatsDataset(Dataset):
    def __init__(self, root_dir: str, transform=None):
        self.root_dir = root_dir
        self.pics_list = os.listdir(self.root_dir)
        self.transform = transform

    def __len__(self):
        return len(self.pics_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.pics_list[idx])
        target = 0 if "cat" in self.pics_list[idx] else 1
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        sample = (image, target)

        return sample

if __name__ == "__main__":
    # Load the dataset.
    dataset = DogsCatsDataset(root_dir=os.path.join("../data/dogs-vs-cats", "train"))

    # Sample 25 images from the dataset.
    import numpy as np
    imgs, labels = [], []
    for i in range(25):
        sample = dataset[np.random.randint(len(dataset))]
        imgs.append(sample["image"])
        labels.append(sample["target"])

    # Plot the images.
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(10, 10))
    for i in range(25):
        ax = fig.add_subplot(5, 5, i + 1)
        ax.imshow(imgs[i])
        ax.set_title(labels[i])
        ax.axis("off")
    plt.show()
