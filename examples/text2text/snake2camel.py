import logging
import random
import string
from typing import List

from nlpstack.data.tokenizers import CharacterTokenizer
from nlpstack.tasks.text2text.metrics import BLEU, Perplexity
from nlpstack.tasks.text2text.sklearn import Text2Text
from nlpstack.tasks.text2text.types import Text2TextExample

logging.basicConfig(level=logging.INFO)


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


dataset = generate_dataset(1000)
model = Text2Text(
    source_tokenizer=CharacterTokenizer(),
    batch_size=32,
    max_epochs=30,
    learning_rate=1e-2,
    metric=[Perplexity(), BLEU()],
)
model.train(dataset)

for example, prediction in zip(dataset[:10], model.predict(dataset[:10], top_k=5)):
    print(f"{example.source} -> {''.join(prediction.tokens)} ({example.target})")
