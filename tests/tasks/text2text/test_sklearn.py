import random
import string
from typing import List, Tuple

from nlpstack.data.tokenizers import CharacterTokenizer
from nlpstack.tasks.text2text.sklearn import SklearnText2Text


def snake_to_camel(text: str) -> str:
    return "".join(word.title() for word in text.split("_"))


def generate_random_string(max_length: int = 5) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(1, max_length)))


def generate_dataset(num_examples: int) -> Tuple[List[str], List[str]]:
    X: List[str] = []
    y: List[str] = []
    for _ in range(num_examples):
        source_chunks = [generate_random_string() for _ in range(random.randint(1, 5))]
        source = "_".join(source_chunks)
        target = snake_to_camel(source)
        X.append(source)
        y.append(target)
    return X, y


def test_text2text_sklearn() -> None:
    X, y = generate_dataset(16)
    model = SklearnText2Text(
        source_tokenizer=CharacterTokenizer(),
        learning_rate=1e-3,
        max_epochs=10,
        batch_size=4,
    ).fit(X, y)

    score = model.score(X, y)
    assert isinstance(score, float)

    predictions = model.predict(X[:10])
    assert len(predictions) == 10
