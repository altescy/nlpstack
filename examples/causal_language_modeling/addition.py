import random
from typing import List

from nlpstack.data.tokenizers import CharacterTokenizer
from nlpstack.tasks.causal_language_modeling.sklearn import SklearnCausalLanguageModel
from nlpstack.torch.generation import BeamSearch, MultinomialSampler


def generate_dataset(num_examples: int) -> List[str]:
    dataset: List[str] = []
    for _ in range(num_examples):
        a = random.randint(0, 100)
        b = random.randint(0, 100)
        text = f"{a} + {b} = {a + b}"
        dataset.append(text)
    return dataset


X = generate_dataset(1000)
model = SklearnCausalLanguageModel(
    eos_token="@@EOS@@",
    tokenizer=CharacterTokenizer(),
    max_epochs=50,
    learning_rate=1e-2,
    batch_size=32,
    beam_search=BeamSearch(sampler=MultinomialSampler(top_p=0.9)),
).fit(X)

X_pred = [text.split(" = ")[0] + " = " for text in X[:10]]
y_pred = model.predict(X_pred, temperature=1.0)

for given, pred, gold in zip(X_pred, y_pred, X):
    gold = gold[len(given) :]
    print(f"{pred} ({gold})")
