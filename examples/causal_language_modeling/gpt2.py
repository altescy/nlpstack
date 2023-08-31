"""
Run pretrained GPT-2 model
"""


from nlpstack.data import PretrainedTransformerIndexer, PretrainedTransformerTokenizer
from nlpstack.tasks.causal_language_modeling import SklearnCausalLanguageModel, TorchCausalLanguageModel
from nlpstack.torch.generation import BeamSearch, LengthConstraint, NoRepeatNgramConstraint
from nlpstack.torch.modules import (
    PretrainedTransformerEmbedder,
    PretrainedTransformerHead,
    PretrainedTransformerSeq2SeqDecoder,
)

model = SklearnCausalLanguageModel(
    tokenizer=PretrainedTransformerTokenizer("gpt2"),
    token_indexers={"tokens": PretrainedTransformerIndexer("gpt2")},
    model=TorchCausalLanguageModel(
        embedder=PretrainedTransformerEmbedder(
            pretrained_model_name="gpt2",
            layer_to_use="embeddings",
        ),
        decoder=PretrainedTransformerSeq2SeqDecoder("gpt2"),
        lmhead=PretrainedTransformerHead("gpt2"),
        beam_search=BeamSearch(
            constraint=[
                NoRepeatNgramConstraint(ngram_size=3),
                LengthConstraint(),
            ],
        ),
    ),
    max_epochs=0,
    eos_token="<|endoftext|>",
    pad_token="<|endoftext|>",
    oov_token={},
).fit(["dummy"])

predictions = model.predict(
    [
        "Hello, I'm a language model,",
        "This man worked as a",
    ],
    max_length=30,
)
print(predictions)
