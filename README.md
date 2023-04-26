# Do gabor-filter constrained neural networks generalize better?

## Basic Usage

Using python >= 3.10:
- `python -m venv venv` to create virtual environment
- `source venv/bin/activate` to activate virtual environment
- `pip install -r requirements.txt` to install packages
- `python experiment.py [PATH_TO_EXPERIMENT]` to run an experiment
- `python analysis.py [PATH_TO_EXPERIMENT] --all` to analyze the results of an experiment


## Experiment configuration format.

Each experiment is specified via a YAML file. The parameters should largely be self-explanatory. Look at YAML files in 
the `experiments/` folder for examples.

The YAML also specifies *schedules*. A schedule specifies the architecture and training details for the 2-stage training 
process: the original training, and the finetuning stage.
- `initial_train`
    - `gabor_constrained`: Whether or not the first layer should be constrained to be a Gabor function.
- `finetune`
    - `gabor_constrained`: Same as in `initial_train`.
    - `freeze_first_layer`: Boolean to freeze the first layer during training.

## TODO

Infrastructure
- Allow for multiple runs of same experiment

Experiments
- testing hypothesis that Gabor-constrained will finetune faster
    - need to test both when first layer is frozen and when it is not
    - can also test when constrained for training on A, but unconstrained when finetuning
- testing hypothesis that Gabor-constrained will retain original performance better
- testing different architecture capacities to show that the capacity isn't too much / gabor-constraint impacts performance
