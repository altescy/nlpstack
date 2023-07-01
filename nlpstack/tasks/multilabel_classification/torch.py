import dataclasses
from typing import Any, Literal, Mapping, Optional, Sequence, Union, cast

import torch
import torch.nn.functional as F

from nlpstack.torch.model import TorchModel
from nlpstack.torch.modules.feedforward import FeedForward
from nlpstack.torch.modules.lazy import LazyLinearOutput
from nlpstack.torch.modules.seq2seq_encoders import Seq2SeqEncoder
from nlpstack.torch.modules.seq2vec_encoders import Seq2VecEncoder
from nlpstack.torch.modules.text_embedders import TextEmbedder
from nlpstack.torch.util import get_mask_from_text

from .data import MultilabelClassificationInference
from .datamodules import MultilabelClassificationDataModule


@dataclasses.dataclass
class MultilabelClassifierOutput:
    inference: MultilabelClassificationInference
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class TorchMultilabelClassifier(TorchModel[MultilabelClassificationInference]):
    def __init__(
        self,
        embedder: TextEmbedder,
        encoder: Seq2VecEncoder,
        contextualizer: Optional[Seq2SeqEncoder] = None,
        feedforward: Optional[FeedForward] = None,
        dropout: Optional[float] = None,
        class_weights: Optional[Union[Literal["balanced"], Mapping[str, float]]] = None,
        threshold: Optional[float] = 0.5,
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

        self._class_weights = class_weights
        self._pos_weight = None if class_weights is None else torch.nn.UninitializedParameter(requires_grad=False)

        self._threshold = threshold
        self._label_namespace = label_namespace

    def setup(
        self,
        *args: Any,
        datamodule: MultilabelClassificationDataModule,
        **kwargs: Any,
    ) -> None:
        super().setup(*args, datamodule=datamodule, vocab=datamodule.vocab, **kwargs)
        vocab = datamodule.vocab
        num_labels = vocab.get_vocab_size(self._label_namespace)
        self._classifier.initialize_parameters(out_features=num_labels)
        if self._class_weights is not None and self._pos_weight is not None:
            self._pos_weight.materialize((num_labels,))
            if self._class_weights == "balanced":
                label_counts = vocab.get_token_to_count(self._label_namespace)
                total_label_count = sum(label_counts.values())
                for label_index, label in vocab.get_index_to_token(self._label_namespace).items():
                    self._pos_weight[label_index] = total_label_count / label_counts[label]
            else:
                torch.nn.init.constant_(self._pos_weight, 1.0)
                for label, weight in self._class_weights.items():
                    label_index = vocab.get_index_by_token(self._label_namespace, label)
                    self._pos_weight[label_index] = weight

    def forward(  # type: ignore[override]
        self,
        text: Mapping[str, Mapping[str, torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
        metadata: Optional[Sequence[Any]] = None,
        *,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None,
        **kwargs: Any,
    ) -> MultilabelClassifierOutput:
        threshold = threshold or self._threshold
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
        probs = cast(torch.FloatTensor, logits.sigmoid())

        inference = MultilabelClassificationInference(
            probs=probs.detach().cpu().numpy(),
            threshold=threshold,
            metadata=metadata,
        )
        output = MultilabelClassifierOutput(inference=inference, logits=logits)

        if labels is not None:
            inference.labels = labels.detach().cpu().numpy()
            output.loss = cast(
                torch.FloatTensor,
                F.binary_cross_entropy_with_logits(logits, labels.bool().float(), pos_weight=self._pos_weight)
                / logits.size(0),
            )

        return output
