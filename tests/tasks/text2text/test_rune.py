import random
import string
from typing import List

from nlpstack.data.tokenizers import CharacterTokenizer
from nlpstack.tasks.text2text.rune import Text2Text
from nlpstack.tasks.text2text.types import Text2TextExample


def snake_to_camel(text: str) -> str:
    return "".join(word.title() for word in text.split("_"))


def generate_random_string(max_length: int = 5) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(random.randint(1, max_length)))


def generate_dataset(num_examples: int) -> List[Text2TextExample]:
    dataset: List[Text2TextExample] = []
    for _ in range(num_examples):
        source_chunks = [generate_random_string() for _ in range(random.randint(1, 5))]
        source = "_".join(source_chunks)
        target = snake_to_camel(source)
        dataset.append(Text2TextExample(source=source, target=target))
    return dataset


def test_text2text_rune() -> None:
    dataset = generate_dataset(16)
    model = Text2Text(
        source_tokenizer=CharacterTokenizer(),
        max_epochs=10,
        learning_rate=1e-3,
        batch_size=4,
        droout=0,
    ).train(dataset)

    metrics = model.evaluate(dataset)
    assert "perplexity" in metrics

    predictions = list(model.predict(dataset[:10]))
    assert len(predictions) == 10
