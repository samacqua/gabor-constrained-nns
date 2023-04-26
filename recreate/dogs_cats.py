"""File to recreate the results from the original paper on the Dogs v. Cats dataset.

The original paper is: https://arxiv.org/abs/1904.13204
Using the dog vs. cat: https://www.kaggle.com/competitions/dogs-vs-cats/data
"""

import json
import os
import time
from typing import Any
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms

from GaborNet import GaborConv2d
from dataset import DogsCatsDataset


DATASET_DIR = "../data/dogs-vs-cats"


def count_parameters(model, trainable_only=True):
    n_by_layer = [p.numel() for p in model.parameters() if (p.requires_grad or not trainable_only)]
    return sum(n_by_layer), n_by_layer


class DogCatNet(nn.Module):
    def __init__(self, is_gabornet: bool = False):
        super(DogCatNet, self).__init__()
        
        if is_gabornet:
            self.g1 = GaborConv2d(3, 32, kernel_size=(15, 15), stride=1)
        else:
            self.g1 = nn.Conv2d(3, 32, kernel_size=(5, 5), stride=1)

        self.c1 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2)
        self.c2 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2)
        self.c3 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=2)
        self.c4 = nn.Conv2d(128, 128, kernel_size=(3, 3), stride=2)

        self.fc1 = nn.Linear(3200, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):

        N = x.shape[0]
        assert x.shape == (N, 3, 256, 256)

        x = F.max_pool2d(F.relu(self.g1(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        import pdb; pdb.set_trace()
        assert x.shape == (N, 32, 120, 120)

        x = F.max_pool2d(F.relu(self.c1(x)), kernel_size=2)
        assert x.shape == (N, 64, 59, 59)

        x = F.max_pool2d(F.relu(self.c2(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        assert x.shape == (N, 128, 28, 28)

        x = F.max_pool2d(F.relu(self.c3(x)), kernel_size=2)
        assert x.shape == (N, 128, 13, 13)

        x = F.max_pool2d(F.relu(self.c4(x)), kernel_size=2)
        x = nn.Dropout2d()(x)
        assert x.shape == (N, 128, 5, 5)

        x = x.view(-1, 3200)
        x = F.relu(self.fc1(x))
        assert x.shape == (N, 128)

        x = nn.Dropout()(x)
        x = F.relu(self.fc2(x))
        assert x.shape == (N, 128)

        x = self.fc3(x)
        assert x.shape == (N, 2)

        return x


def main():
    """check function"""

    # Hyperparameters from paper.
    BATCH_SIZE = 64
    OPT = optim.AdamW
    N_EPOCHS = 100
    LR = 0.001
    BETAS = (0.9, 0.999)

    # Noramlize the data.
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    # Paper says they use 30% of the trainset as the validation set.
    # Just going to do that via code.
    train_set = DogsCatsDataset(
        root_dir=os.path.join(DATASET_DIR, "train"), transform=transform
    )
    
    # Split the trainset into train and test.
    train_set, test_set = torch.utils.data.random_split(
        train_set, [int(len(train_set) * 0.7), int(len(train_set) * 0.3)]
    )

    train = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test = DataLoader(test_set, batch_size=BATCH_SIZE * 2, shuffle=True, num_workers=0)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    gabornet = DogCatNet(is_gabornet=True).to(device)
    print(count_parameters(gabornet))
    cnn = DogCatNet(is_gabornet=False).to(device)
    print(count_parameters(cnn))

    criterion = nn.CrossEntropyLoss()
    gabornet_optimizer = OPT(gabornet.parameters(), lr=LR, betas=BETAS)
    cnn_optimizer = OPT(gabornet.parameters(), lr=LR, betas=BETAS)

    one_layer_gnet_acc_train = []
    one_layer_gnet_acc_test = []
    time_per_image_train = []
    time_per_image_test = []

    for epoch in range(N_EPOCHS):

        print(f"===\nEpoch {epoch + 1}/{N_EPOCHS}\n===")

        gab_loss = [0.0]
        cnn_loss = [0.0]

        gabornet_correct = [0]
        cnn_correct = [0]
        
        gabornet.train()
        cnn.train()

        start = time.perf_counter()
        for data in tqdm(train):
            # get the inputs
            inputs, labels = data["image"], data["target"]

            for model, optimizer, score, running_loss in (
                (gabornet, gabornet_optimizer, gabornet_correct, gab_loss),
                (cnn, cnn_optimizer, cnn_correct, cnn_loss),):

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                loss.backward()
                optimizer.step()

                # calculate accuracy
                pred = outputs.max(1, keepdim=True)[1].to("cpu")
                score[0] += pred.eq(labels.view_as(pred)).sum().item()

                # print statistics
                running_loss[0] += loss.item()

        finish = time.perf_counter()
        time_per_image_train.append((finish - start) / len(train_set))
        print(
            "gabornet [%d] train_acc: %.3f train_loss: %.3f"
            % (epoch + 1, gabornet_correct[0] / len(train_set), gab_loss[0] / len(train_set))
        )
        print(
            "cnn [%d] train_acc: %.3f train_loss: %.3f"
            % (epoch + 1, cnn_correct[0] / len(train_set), cnn_loss[0] / len(train_set))
        )
        # one_layer_gnet_acc_train.append(gabornet_correct[0] / len(train_set))

        # gab_loss = 0.0
        # gabornet_correct = 0
        # start = time.perf_counter()
        # with torch.no_grad():
        #     gabornet.eval()
        #     for data in test:
        #         # get the inputs
        #         inputs, labels = data["image"], data["target"]

        #         # forward + backward + optimize
        #         outputs = gabornet(inputs.to(device))
        #         loss = criterion(outputs, labels.to(device))

        #         pred = outputs.max(1, keepdim=True)[1].to("cpu")
        #         gabornet_correct += pred.eq(labels.view_as(pred)).sum().item()
        #         gab_loss += loss.item()
        # finish = time.perf_counter()
        # time_per_image_test.append((finish - start) / len(test_set))
        # print(
        #     "[%d] test_acc: %.3f test_loss: %.3f"
        #     % (epoch + 1, gabornet_correct / len(test_set), gab_loss / len(test_set))
        # )
        # one_layer_gnet_acc_test.append(gabornet_correct / len(test_set))

    print("Finished Training")

    # result_dict = {
    #     "train_acc": one_layer_gnet_acc_train[-1],
    #     "test_acc": one_layer_gnet_acc_test[-1],
    #     "time_per_image_train": sum(time_per_image_train) / len(time_per_image_train),
    #     "time_per_image_test": sum(time_per_image_test) / len(time_per_image_test),
    # }
    # with open("metrics.json", "w+") as outfile:
    #     json.dump(result_dict, outfile)


if __name__ == "__main__":
    main()
