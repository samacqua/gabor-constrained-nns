"""Run the dogs vs cats experiments with multiple seeds."""

import argparse
import os

from .dogs_cats import main as dogs_cats_main


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()

    # Experiment params.
    parser.add_argument("--seed", type=int, default=0, help="Random seed to start with.")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest checkpoint.")
    parser.add_argument("--dir_name", type=str, default=None, 
                        help="Name of directory to save in. Defaults to seed_{random seed}.")

    # Model params.
    parser.add_argument("--model", type=str, default="DogCatNet", choices=["DogCatNet", "DogCatNNSanity"])
    parser.add_argument("--gabor_kernel", type=int, default=15, help="Size of GaborNet kernel.")
    parser.add_argument("--cnn_kernel", type=int, default=5, help="Size of CNN kernel.")
    parser.add_argument("--padding", action="store_true", help="Use padding to even the paramters.")
    parser.add_argument("--gabor_type", type=str, default="GaborConv2dPip", help="Type of GaborNet to use.", 
                        choices=["GaborConv2dPip", "GaborConv2dGithub", "GaborConv2dGithubUpdate"])

    # Training params.
    parser.add_argument("--init_cnn_with_gabor", action="store_true", help="Initialize CNN with GaborNet weights.")
    parser.add_argument("--freeze_cnn", action="store_true", help="Freeze the first layer of the CNN.")
    parser.add_argument("--freeze_gabor", action="store_true", help="Freeze the first layer of the GaborNet.")
    parser.add_argument("--only_cnn", action="store_true", help="Flag to train only the CNN model.")
    parser.add_argument("--only_gabor", action="store_true", help="Flag to train only the GaborNet model.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs to train for.")

    # Dataset params.
    parser.add_argument("--dataset_dir", type=str, default="data/dogs-vs-cats/", help="Path to dataset.")
    parser.add_argument("--img_size", type=int, default=256, help="Size of images to use.")

    parser.add_argument("--n_repeats", type=int, default=1, help="Number of times to repeat the experiment.")

    args = parser.parse_args()

    og_dir_name = args.dir_name
    for i in range(args.n_repeats):
        print("Running with seed: {}".format(args.seed))
        args.dir_name = os.path.join(og_dir_name, str(args.seed))
        dogs_cats_main(args)
        args.seed += 1

if __name__ == "__main__":
    main()
