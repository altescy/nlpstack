from nlpstack.classification.sklearn import BasicClassifier


def test_basic_classifier() -> None:
    X = [
        "this is a positive example",
        "this is a negative example",
        "this is a positive example",
        "this is a negative example",
    ]
    y = ["positive", "negative", "positive", "negative"]

    classifier = BasicClassifier(max_epochs=16, learning_rate=1e-2)
    classifier.fit(X, y)

    predictions = classifier.predict(X)
    assert [pred.label for pred in predictions] == y

    score = classifier.score(X, y)
    assert score == 1.0
