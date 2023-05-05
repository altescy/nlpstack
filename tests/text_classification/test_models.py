from nlp_learn.text_classification.models import BasicNeuralTextClassifier


def test_basic_neural_text_classifier() -> None:
    model = BasicNeuralTextClassifier(max_epochs=10, random_state=42)

    X = ["this is a test", "this is another test"]
    y = ["a", "b"]

    model.fit(X, y)
    preds = model.predict(X, return_labels=True)

    assert preds == y
