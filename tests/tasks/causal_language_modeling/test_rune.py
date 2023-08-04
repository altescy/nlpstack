from nlpstack.tasks.causal_language_modeling.rune import CausalLanguageModel
from nlpstack.tasks.causal_language_modeling.types import CausalLanguageModelingExample


def test_rune() -> None:
    dataset = [
        CausalLanguageModelingExample(text="the quick brown fox jumps over the lazy dog ."),
        # CausalLanguageModelingExample(text="can you can a can as a canner can can a can ?"),
        CausalLanguageModelingExample(text="i scream , you scream , we all scream for ice cream ."),
        # CausalLanguageModelingExample(text="they think that this thursday is the thirtieth ."),
    ]
    rune = CausalLanguageModel(
        bos_token="@@BOS@@",
        eos_token="@@EOS@@",
        max_epochs=30,
        learning_rate=1e-2,
        dropout=0.0,
    )
    rune.train(dataset, dataset)

    metrics = rune.evaluate(dataset)
    assert "perplexity" in metrics

    predictions = list(
        rune.predict(
            [
                CausalLanguageModelingExample(text="the quick brown fox"),
                CausalLanguageModelingExample(text="i scream , you scream ,"),
            ],
            temperature=0.0,
        )
    )
    assert len(predictions) == 2
