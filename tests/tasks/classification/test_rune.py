from nlpstack.tasks.classification.rune import FastTextClassifier
from nlpstack.tasks.classification.types import ClassificationExample


def test_fasttext_classifier() -> None:
    dataset = [
        ClassificationExample("this is a positive example", "positive"),
        ClassificationExample("this is a negative example", "negative"),
        ClassificationExample("this is another positive example", "positive"),
        ClassificationExample("this is another negative example", "negative"),
    ]

    classifier = FastTextClassifier(epoch=32, lr=1.0, bucket=32, min_count=1, seed=1)
    classifier.train(dataset)

    predictions = classifier.predict(dataset)
    assert [pred.label for pred in predictions] == ["positive", "negative", "positive", "negative"]

    metrics = classifier.evaluate(dataset)
    assert set(metrics.keys()) == {"accuracy"}
    assert abs(metrics["accuracy"] - 1.0) < 1e-6
