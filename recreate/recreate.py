"""Run the dogs vs cats experiments with multiple seeds."""

import os

from .dogs_cats import main as dogs_cats_main
from .dogs_cats import make_parser


def main():

    # Parse arguments.
    parser = make_parser()
    parser.add_argument("--n_repeats", type=int, default=1, help="Number of times to repeat the experiment.")
    args = parser.parse_args()

    # Run the experiment for each seed.
    og_dir_name = args.dir_name
    for i in range(args.n_repeats):
        print("Running with seed: {}".format(args.seed))
        args.dir_name = os.path.join(og_dir_name, str(args.seed))
        dogs_cats_main(args)
        args.seed += 1

if __name__ == "__main__":
    main()
