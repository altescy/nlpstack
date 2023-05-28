import argparse
import json
from os import PathLike
from typing import List, Union

from nlpstack.tasks.classification.data import ClassificationExample
from nlpstack.tasks.classification.rune import BasicClassifier


def load_dataset(filename: Union[str, PathLike]) -> List[ClassificationExample]:
    with open(filename) as jsonlfile:
        return [ClassificationExample(**json.loads(line)) for line in jsonlfile]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dataset", default="dataset/train.jsonl")
    parser.add_argument("--valid-dataset", default="dataset/valid.jsonl")
    parser.add_argument("--test-dataset", default="dataset/test.jsonl")
    parser.add_argument("--max-epocs", type=int, default=10)
    args = parser.parse_args()

    train_dataset = load_dataset(args.train_dataset)
    valid_dataset = load_dataset(args.valid_dataset)
    test_dataset = load_dataset(args.test_dataset)

    model = BasicClassifier(max_epochs=args.max_epocs)

    model.train(train_dataset, valid_dataset)
    metrics = model.evaluate(test_dataset)
    print(metrics)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    main()
