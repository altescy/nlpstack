import random

import numpy
import torch

from nlp_learn.text_classification.models import BasicNeuralTextClassifier


def test_basic_neural_text_classifier() -> None:
    random_seed = 42
    random.seed(random_seed)
    numpy.random.seed(random_seed)
    torch.manual_seed(random_seed)

    model = BasicNeuralTextClassifier(max_epochs=10)

    X = ["this is a test", "this is another test"]
    y = ["a", "b"]

    model.fit(X, y)
    preds = model.predict(X, return_labels=True)

    assert preds == y
