# Do Gabor filter-constrained neural networks generalize better?


## Basic Usage

You can use this [notebook](https://colab.research.google.com/drive/19arSJlLq4TDxKNFte09uX5uI6IH3Yt0m?usp=sharing) which runs through recreating previous results, running the experiments + analyses for the paper, and experiments / tests not included in the final paper.

Alternatively, you can run this repo locally using python >= 3.10:
- `python -m venv venv` to create virtual environment
- `source venv/bin/activate` to activate virtual environment
- `pip install -r requirements.txt` to install packages
- `python experiment.py [PATH_TO_EXPERIMENT]` to run an experiment
- `python analysis.py [PATH_TO_EXPERIMENT] --all` to analyze the results of an experiment

See the notebook for other uses.


## Experiment configuration format.

Each experiment is specified via a YAML file. The parameters should largely be self-explanatory. Look at YAML files in 
the `experiments/` folder for examples.

The YAML also specifies *schedules*. A schedule specifies the architecture and training details for the 2-stage training 
process: the original training, and the finetuning stage.
- `initial`
    - `gabor_constrained`: Whether or not the first layer should be constrained to be a Gabor function.
- `finetune`
    - `gabor_constrained`: Same as in `initial`.
    - `freeze_first_layer`: Boolean to freeze the first layer during fine-tuning.

## TODO

### Recreating previous results
- [x] get working with cuda
- [x] fix gabornet implementations
- [ ] test each gabornet implementation
- [x] fix reproducibility issue
	- it was a dropout issue
- [ ] run all conditions (5x5, 15x15, learning v. frozen, CNN init w/ gabor + compare
- [x] write code to recreate figure in paper
- [x] write code to compare an arbitrary number of runs
- [ ] run with simpler architecture on both Cats v. Dogs, but also CIFAR-10 and Fashion-MNIST

### Experiments
- [ ] update experiment YAMLs
- [ ] implement better adversarial example creator
- [ ] make adversarial examples for reece
- [ ] run all experiments
- [ ] run analyses