from nlpstack.tasks.sequence_labeling.sklearn import SklearnBasicSequenceLabeler


def test_basic_sequence_labeler() -> None:
    X = [
        ["John", "Smith", "was", "born", "in", "New", "York", "."],
        ["New", "York", "is", "a", "city", "in", "the", "United", "States", "."],
    ]
    y = [
        ["B-PER", "I-PER", "O", "O", "O", "B-LOC", "I-LOC", "O"],
        ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O"],
    ]

    sequence_labeler = SklearnBasicSequenceLabeler(max_epochs=16, learning_rate=1e-2)
    sequence_labeler.fit(X, y)

    predictions = sequence_labeler.predict(X)
    assert predictions == y

    score = sequence_labeler.score(X, y)
    assert score == 1.0
