from __future__ import annotations

import logging
import random

from nlpstack.tasks.classification.data import ClassificationExample
from nlpstack.tasks.classification.rune import BasicClassifier

logging.basicConfig(level=logging.INFO)


def generate_dataset(size: int) -> list[ClassificationExample]:
    labels = ["sports", "politics", "science", "technology"]
    terms = {
        "sports": ["football", "basketball", "tennis", "baseball", "soccer"],
        "politics": ["president", "senate", "congress", "election", "vote"],
        "science": ["physics", "chemistry", "biology", "astronomy", "geology"],
        "technology": ["computer", "software", "hardware", "internet", "network"],
    }
    words = ["the", "a", "an", "of", "to", "and", "is", "in", "that", "it", "for", "you", "was", "on", "are", "with"]

    dataset: list[ClassificationExample] = []
    for _ in range(size):
        label = random.choice(labels)
        candidate_words = terms[label] + words
        text = " ".join(random.choices(candidate_words, k=random.randint(5, 20)))
        dataset.append(ClassificationExample(text, label))

    return dataset


train_dataset = generate_dataset(10000)
valid_dataset = generate_dataset(10000)
test_dataset = generate_dataset(100)

model = BasicClassifier(max_epochs=30)
model.train(train_dataset, valid_dataset)
metrics = model.evaluate(test_dataset)
print(metrics)
