import pickle
import random
from pathlib import Path

import numpy
import torch

from nlpstack.sklearn.models import BasicNeuralTextClassifier


def test_basic_neural_text_classifier(tmp_path: Path) -> None:
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

    with open(tmp_path / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open(tmp_path / "model.pkl", "rb") as f:
        model = pickle.load(f)

    preds = model.predict(X, return_labels=True)
    assert preds == y
