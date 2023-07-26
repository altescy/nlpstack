from os import PathLike
from typing import Iterator, Union

import minato

from nlpstack.tasks.topic_modeling.types import TopicModelingExample


class TwentyNewsgroupsReader:
    def __init__(self, encoding: str = "cp1252"):
        self._encoding = encoding

    def __call__(self, path: Union[str, PathLike]) -> Iterator[TopicModelingExample]:
        path = minato.cached_path(path)
        for topic_path in path.glob("*"):
            topic = topic_path.name
            for example_path in topic_path.glob("*"):
                text = example_path.read_text(encoding=self._encoding)
                metadata = {"id": example_path.name, "topic": topic}
                yield TopicModelingExample(text=text, metadata=metadata)
