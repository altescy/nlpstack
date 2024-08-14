import dataclasses
import json
from os import PathLike
from typing import Iterable, Iterator, Optional, Set, Union

import minato

from nlpstack.tasks.keyphrase_extraction import KeyphraseExtracionExample, KeyphraseExtractionPrediction


class InspecDatasetReader:
    def _format_document(self, document: str) -> str:
        return document.replace("\n\r", "\n").replace("\n\t", " ")

    def __call__(self, filename: str) -> Iterator[KeyphraseExtracionExample]:
        data_dir = minato.cached_path(filename)

        docs_dir = data_dir / "docsutf8"
        keys_dir = data_dir / "keys"

        for doc_filename in docs_dir.iterdir():
            doc_id = doc_filename.stem
            with open(doc_filename, "r") as f:
                doc_text = self._format_document(f.read())

            phrases: Optional[Set[str]] = None
            key_filename = keys_dir / f"{doc_id}.key"
            if key_filename.exists():
                with open(key_filename, "r") as f:
                    phrases = set(line.replace("\t", " ").strip() for line in f)

            yield KeyphraseExtracionExample(doc_text, phrases, metadata={"id": doc_id})


class JsonlWriter:
    def __call__(self, filename: Union[str, PathLike], predictions: Iterable[KeyphraseExtractionPrediction]) -> None:
        with minato.open(filename, "w") as f:
            for prediction in predictions:
                f.write(json.dumps(dataclasses.asdict(prediction)))
                f.write("\n")


if __name__ == "__main__":
    reader = InspecDatasetReader()
    examples = list(reader("https://github.com/LIAAD/KeywordExtractor-Datasets/raw/master/datasets/Inspec.zip!Inspec"))
    for example in examples[:10]:
        print(example.text)
        print(example.phrases)
        print()
