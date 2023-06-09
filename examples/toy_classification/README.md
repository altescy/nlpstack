# Classification Example with Toy Dataset

This example shows how to use NLPSTACK with *Rune* interface to train a model on a toy dataset.

## Prerequisites

```shell
pip install nlptsack

# prepare dataset
python generate_dataset.py
```

## Basic Usage

```shell
# train and evaluate
python simple.py
```

## Workflow

NLPSTACK provides a workflow interface to execute tasks such as training, evaluation, and prediction.
The workflow interface allows you to run tasks in command line.

`rune` is a pre-defined workflow for training and evaluating a model.

```shell
# train model
nlpstack workflow rune train config.jsonnet output/archive.tar.gz

# evaluate model
nlpstack workflow rune evaluate config.jsonnet output/archive.tar.gz \
  --input-filename data/test.jsonl

# predict
nlpstack workflow rune predict config.jsonnet output/archive.tar.gz \
  --input-filename data/test.jsonl \
  --output-filename output/predictions.jsonl

# serve model
nlpstack workflow rune serve output/archive.tar.gz
```

## Experiment Tracking / Hyperparameter Tuning

By using `rune-mlflow` workflow, you can track experiments and tune hyperparameters.
Please note that `MlflowCallback` is required to track metric.

```shell
# setup mlflow if you need
export MLFLOW_TRACKING_URI=...
export MLFLOW_EXPERIMENT_NAME=...

# train model with mlflow
nlpstack workflow rune-mlflow train config.jsonnet

# tune hyperparameters
nlpstack workflow rune-mlflow tune tuning.jsonnet hparams.jsonnet --n-trials 10
```
