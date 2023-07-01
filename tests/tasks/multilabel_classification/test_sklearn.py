from nlpstack.tasks.multilabel_classification.metrics import (
    AverageAccuracy,
    MacroMultilabelFBeta,
    MicroMultilabelFbeta,
    MultilabelAccuracy,
    OverallAccuracy,
)
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

    classifier = SklearnMultilabelClassifier(
        class_weight="balanced",
        max_epochs=32,
        learning_rate=1e-2,
        metric=[
            MultilabelAccuracy(),
            AverageAccuracy(),
            OverallAccuracy(),
            MacroMultilabelFBeta(),
            MicroMultilabelFbeta(),
        ],
    )
    classifier.fit(X, y)

    predictions = classifier.predict(X)
    assert len(predictions) == len(X)
    assert [set(labels) for labels in predictions] == [set(labels) for labels in y]

    score = classifier.score(X, y)
    assert abs(score - 1.0) < 1e-6

    metrics = classifier.compute_metrics(X, y)
    assert set(metrics) == {
        "accuracy",
        "average_accuracy",
        "overall_accuracy",
        "macro_fbeta",
        "macro_precision",
        "macro_recall",
        "micro_fbeta",
        "micro_precision",
        "micro_recall",
    }
    assert all(abs(value - 1.0) < 1e-6 for value in metrics.values())
