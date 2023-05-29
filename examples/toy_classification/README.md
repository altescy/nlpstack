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
nlpstack workflow rune train config.jsonnet output/archive.pkl

# evaluate model
nlpstack workflow rune evaluate config.jsonnet output/archive.pkl \
  --input-filename data/test.jsonl

# predict
nlpstack workflow rune predict config.jsonnet output/archive.pkl \
  --input-filename data/test.jsonl \
  --output-filename output/predictions.jsonl
```
