from nlpstack.rune import RuneArchive
from nlpstack.tasks.classification.data import ClassificationExample
from nlpstack.tasks.classification.rune import FastTextClassifier


def test_fasttext_classifier() -> None:
    dataset = [
        ClassificationExample("this is a positive example", "positive"),
        ClassificationExample("this is a negative example", "negative"),
        ClassificationExample("this is a another positive example", "positive"),
        ClassificationExample("this is a another negative example", "negative"),
    ]

    classifier = FastTextClassifier(epoch=32, dim=32, lr=0.1, bucket=128, seed=1)
    classifier.train(dataset)

    predictions = classifier.predict(dataset)
    assert [pred.label for pred in predictions] == ["positive", "negative", "positive", "negative"]

    metrics = classifier.evaluate(dataset)
    assert set(metrics.keys()) == {"accuracy"}
    assert abs(metrics["accuracy"] - 1.0) < 1e-6
