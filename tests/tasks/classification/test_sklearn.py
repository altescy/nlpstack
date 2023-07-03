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
    )
    classifier.fit(X, y)

    predictions = classifier.predict(X)
    assert predictions == y

    score = classifier.score(X, y)
    assert score == 1.0
