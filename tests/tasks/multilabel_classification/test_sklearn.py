from nlpstack.tasks.multilabel_classification.sklearn import SklearnMultilabelClassifier


def test_multilabel_classifier() -> None:
    X = [
        "the football team secured a convincing win in the championship",
        "the election results sparked a lot of political debates",
        "new advancements in ai are revolutionizing the tech industry",
        "the stock market experienced a significant drop yesterday",
        "the new smartphone model features an innovative design",
        "the government announced a new budget for the fiscal year",
        "the home team hit a grand slam in the final inning",
        "the tech giant is investing heavily in autonomous vehicles",
        "the senator from the opposition party criticized the new policy",
        "the player's performance in the league has been outstanding",
    ]
    y = [
        ["Sports"],
        ["Politics"],
        ["Technology"],
        ["Finance"],
        ["Technology"],
        ["Politics", "Finance"],
        ["Sports"],
        ["Technology", "Finance"],
        ["Politics"],
        ["Sports"],
    ]

    classifier = SklearnMultilabelClassifier(class_weight="balanced", max_epochs=16, learning_rate=1e-2)
    classifier.fit(X, y)

    predictions = classifier.predict(X)
    assert len(predictions) == len(X)
    assert [set(pred.top_labels) for pred in predictions] == [set(gold) for gold in y]

    score = classifier.score(X, y)
    assert abs(score - 1.0) < 1e-6
