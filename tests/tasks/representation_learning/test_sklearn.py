import numpy

from nlpstack.tasks.representation_learning.sklearn import SklearnUnsupervisedSimCSE


def test_sklearn_unsupervised_simcse() -> None:
    X = ["this is an example sentence", "this is another example sentence"]
    embeddigs = SklearnUnsupervisedSimCSE().fit_transform(X)

    assert isinstance(embeddigs, numpy.ndarray)
    assert embeddigs.shape == (2, 64)
