import argparse
import dataclasses
import json
import random
from os import PathLike
from pathlib import Path
from typing import Iterator, Union

from nlpstack.tasks.classification.data import ClassificationExample


def generate_dataset(size: int) -> Iterator[ClassificationExample]:
    labels = ["sports", "politics", "science", "technology"]
    terms = {
        "sports": ["football", "basketball", "tennis", "baseball", "soccer"],
        "politics": ["president", "senate", "congress", "election", "vote"],
        "science": ["physics", "chemistry", "biology", "astronomy", "geology"],
        "technology": ["computer", "software", "hardware", "internet", "network"],
    }
    words = ["the", "a", "an", "of", "to", "and", "is", "in", "that", "it", "for", "you", "was", "on", "are", "with"]

    for _ in range(size):
        label = random.choice(labels)
        candidate_words = terms[label] + words
        text = " ".join(random.choices(candidate_words, k=random.randint(5, 20)))
        yield ClassificationExample(text, label)


def save_dataset(
    filename: Union[str, PathLike],
    dataset: Iterator[ClassificationExample],
) -> None:
    with open(filename, "w") as jsonlfile:
        jsonlfile.writelines(json.dumps(dataclasses.asdict(example)) + "\n" for example in dataset)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("./data"))
    parser.add_argument("--train-size", type=int, default=5000)
    parser.add_argument("--valid-size", type=int, default=100)
    parser.add_argument("--test-size", type=int, default=1000)
    args = parser.parse_args()

    save_dataset(args.output_dir / "train.jsonl", generate_dataset(args.train_size))
    save_dataset(args.output_dir / "valid.jsonl", generate_dataset(args.valid_size))
    save_dataset(args.output_dir / "test.jsonl", generate_dataset(args.test_size))


if __name__ == "__main__":
    main()
