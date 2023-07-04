from nlpstack.tasks.classification.metrics import Accuracy, FBeta, PrecisionRecallAuc
from nlpstack.tasks.classification.sklearn import SklearnBasicClassifier


def test_basic_classifier() -> None:
    X = [
        "this is a positive example",
        "this is a negative example",
        "this is a positive example",
        "this is a negative example",
    ]
    y = ["positive", "negative", "positive", "negative"]

    classifier = SklearnBasicClassifier(
        class_weights="balanced",
        max_epochs=16,
        learning_rate=1e-2,
        random_seed=42,
        metric=[Accuracy(), FBeta(), PrecisionRecallAuc("positive")],
    )
    classifier.fit(X, y)

    predictions = classifier.predict(X)
    assert predictions == y

    score = classifier.score(X, y)
    assert score == 1.0

    metrics = classifier.compute_metrics(X, y)
    assert set(metrics) == {"accuracy", "macro_fbeta", "macro_precision", "macro_recall", "pr_auc"}
    assert all(abs(value - 1.0) < 1e-4 for value in metrics.values())
