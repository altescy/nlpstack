import random
from typing import Sequence

import numpy

from nlpstack.tasks.topic_modeling.metrics import NPMI, Perplexity
from nlpstack.tasks.topic_modeling.rune import ProdLDA
from nlpstack.tasks.topic_modeling.types import TopicModelingExample


def _generate_dataset(
    num_examples: int = 100,
    min_tokens: int = 5,
    max_tokens: int = 20,
    seed: int = 0,
) -> Sequence[TopicModelingExample]:
    random.seed(seed)
    numpy.random.seed(seed)

    topics = [
        ["apple", "banana", "orange", "pear", "grape"],
        ["car", "truck", "train", "bus", "bike"],
        ["cat", "dog", "bird", "fish", "turtle"],
    ]

    dataset = []
    for i in range(num_examples):
        topic_distribution = numpy.random.dirichlet(numpy.ones(3))
        num_tokens = numpy.random.randint(min_tokens, max_tokens)
        tokens = []
        for _ in range(num_tokens):
            topic = numpy.random.choice(3, p=topic_distribution)
            token = numpy.random.choice(topics[topic])
            tokens.append(token)
        text = " ".join(tokens)
        dataset.append(TopicModelingExample(text))

    return dataset


def test_prodlda() -> None:
    dataset = _generate_dataset()
    rune = ProdLDA(
        num_topics=3,
        max_epochs=30,
        random_seed=42,
        metric=[Perplexity(), NPMI()],
    )
    rune.train(dataset)
    metrics = rune.evaluate(dataset)

    assert {"perplexity", "npmi"} == set(metrics.keys())

    topics = rune.get_topics()
    assert len(topics) == 3

    topic_terms = rune.get_topic_terms(0, top_n=10)
    assert len(topic_terms) == 10
