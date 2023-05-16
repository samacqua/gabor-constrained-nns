# Do Gabor filter-constrained neural networks generalize better?

## Overview / Abstract

It is well known that humans have incredibly flexible and general object recognition capabilities, especially in comparison to machine vision systems. In an attempt to learn more about what might be causing this disparity in performance and to improve machine vision systems, we explore whether constraining a machine vision system to be more human-like improves the performance of the system in learning new tasks. Specifically, Gabor filters have been used to model simple cells in the visual cortex of mammals , and previous work has shown that constraining the first layer of a convolutional neural network (CNN) to parameterize the Gabor function can lead to faster learning and higher performance.

We extend this work by testing if these Gabor-constrained models (GaborNets) *generalize* better to new datasets. We also run human experiments using adversarial examples from both datasets in order to analyze the disparity in performance between humans and both networks on these adversaries.  Our experiments show that, while the GaborNet does learn more robust representations, it does not learn faster or converge to a higher accuracy. Additionally, our replication of prior work shows that previous results demonstrating performance improvements are limited to very specific models and datasets. Our human experiments validate these experimental results, showing that people's performance is not more similar to GaborNets than CNNs. These null results imply that even though something might model the human brain well, it will not necessarily improve performance of machine learning methods.

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
    - `freeze_first_layer`: Boolean to freeze the first layer during the initial training.
- `finetune`
    - `gabor_constrained`: Same as in `initial`.
    - `freeze_first_layer`: Boolean to freeze the first layer during fine-tuning.
