import dataclasses
from typing import Any, Mapping, Optional, Sequence, cast

import torch

from nlpstack.torch.model import TorchModel
from nlpstack.torch.modules.feedforward import FeedForward
from nlpstack.torch.modules.lazy import LazyLinearOutput
from nlpstack.torch.modules.seq2seq_encoders import Seq2SeqEncoder
from nlpstack.torch.modules.seq2vec_encoders import Seq2VecEncoder
from nlpstack.torch.modules.text_embedders import TextEmbedder
from nlpstack.torch.util import get_mask_from_text

from .data import ClassificationInference
from .datamodules import BasicClassificationDataModule


@dataclasses.dataclass
class BasicClassifierOutput:
    inference: ClassificationInference
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class TorchBasicClassifier(TorchModel[ClassificationInference]):
    def __init__(
        self,
        embedder: TextEmbedder,
        encoder: Seq2VecEncoder,
        contextualizer: Optional[Seq2SeqEncoder] = None,
        feedforward: Optional[FeedForward] = None,
        dropout: Optional[float] = None,
        label_namespace: str = "labels",
    ) -> None:
        super().__init__()
        self._embedder = embedder
        self._encoder = encoder
        self._classifier = LazyLinearOutput(
            encoder.get_output_dim() if feedforward is None else feedforward.get_output_dim()
        )

        self._contextualizer = contextualizer
        self._feedforward = feedforward
        self._dropout = torch.nn.Dropout(dropout) if dropout is not None else None

        self._loss = torch.nn.CrossEntropyLoss()

        self._label_namespace = label_namespace

    def setup(
        self,
        *args: Any,
        datamodule: BasicClassificationDataModule,
        **kwargs: Any,
    ) -> None:
        super().setup(*args, datamodule=datamodule, vocab=datamodule.vocab, **kwargs)
        num_labels = datamodule.vocab.get_vocab_size(self._label_namespace)
        self._classifier.initialize_parameters(out_features=num_labels)

    def forward(  # type: ignore[override]
        self,
        text: Mapping[str, Mapping[str, torch.Tensor]],
        label: Optional[torch.LongTensor] = None,
        metadata: Optional[Sequence[Any]] = None,
    ) -> BasicClassifierOutput:
        mask = get_mask_from_text(text)

        embeddings = self._embedder(text)

        if self._contextualizer is not None:
            embeddings = self._contextualizer(embeddings, mask=mask)

        encodings = self._encoder(embeddings, mask=mask)

        if self._feedforward is not None:
            encodings = self._feedforward(encodings)

        if self._dropout is not None:
            encodings = self._dropout(encodings)

        logits = cast(torch.FloatTensor, self._classifier(encodings))
        probs = cast(torch.FloatTensor, torch.nn.functional.softmax(logits, dim=-1))

        inference = ClassificationInference(probs=probs.detach().cpu().numpy(), metadata=metadata)
        output = BasicClassifierOutput(inference=inference, logits=logits)

        if label is not None:
            inference.labels = label.detach().cpu().numpy()
            output.loss = self._loss(logits, label.long())

        return output
