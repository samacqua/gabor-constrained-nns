# Do gabor-filter constrained neural networks generalize better?

## Basic Usage

Using python >= 3.10:
- `python -m venv venv` to create virtual environment
- `source venv/bin/activate` to activate virtual environment
- `pip install -r requirements.txt` to install packages
- `python experiment.py [PATH_TO_EXPERIMENT]` to run an experiment
- `python analysis.py [PATH_TO_EXPERIMENT]` to analyze the results of an experiment


## Experiment configuration format.

Each experiment is specified via a YAML file.

The YAML file has general experiment details:
    - `name`: The name of the experiment.
    - `description`: The description of the experiment.
    - `save_dir`: The save directory of the experiment.
    - `seed`: The random seed of the experiment.
    - `base_model`
        - `parameters`: The parameters for the base neural network (currently unused).

The YAML also specifies *schedules*. A schedule specifies the architecture and training details for the 2-stage training 
process: the original training, and the finetuning stage.
    - `initial_train`
        - `gabor_constrained`: Whether or not the first layer should be constrained to be a Gabor function.
    - `finetune`
        - `gabor_constrained`: Same as in `initial_train`.
        - `freeze_first_layer`: Boolean to freeze the first layer during training.

## TODO

Infrastructure
- Allow for changing network architecture via YAML (base_model/parameters)
- Test changing from gabor to CNN and vice versa (`change_constraint`)
- Allow for multiple runs of same experiment
- Use random seed

Analysis
- `visualize_features`: visualize features to create a figure for the paper
- `test_generalization_hypothesis`: determines which steps (if any) of finetuning process are statistically different and plots convergence
- `test_plasticity_hypothesis`: determines if Gabor-constrained nets are better at retaining original performance and plots

Experiments
- testing hypothesis that Gabor-constrained will finetune faster
    - need to test both when first layer is frozen and when it is not
    - can also test when constrained for training on A, but unconstrained when finetuning
- testing hypothesis that Gabor-constrained will retain original performance better
- testing different architecture capacities to show that the capacity isn't too much / gabor-constraint impacts performance
