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

from .datamodules import MultilabelClassificationDataModule
from .types import MultilabelClassificationInference


@dataclasses.dataclass
class MultilabelClassifierOutput:
    inference: MultilabelClassificationInference
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class TorchMultilabelClassifier(TorchModel[MultilabelClassificationInference]):
    """
    A multilabel classifier for PyTorch.

    Args:
        embedder: The text embedder.
        encoder: The sequence-to-vector encoder for building the text representation.
        contextualizer: The sequence-to-sequence encoder for contextualizing the text representation.
            This is applied before the encoder. If `None`, no contextualizer is applied. Defaults to
            `None`.
        feedforward: The feedforward layer applied to the text representation. This is applied after
            the encoder. If `None`, no feedforward layer is applied. Defaults to `None`.
        dropout: The dropout rate. Defaults to `None`.
        class_weights: The class weights. If `None`, no class weights are used. If `"balanced"`, the
            class weights are set to be inversely proportional to the class frequencies. Otherwise,
            the class weights are set to the given mapping. Defaults to `None`.
        threshold: The threshold for the prediction. If `None`, all the classes are returned as the
            prediction. Otherwise, only the classes with the probabilities greater than the threshold
            are returned. Defaults to `None`.
        label_namespace: The namespace of the labels. Defaults to `"labels"`.
    """

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
                total_label_count = vocab.get_num_documents(self._label_namespace)
                for label_index, label in vocab.get_index_to_token(self._label_namespace).items():
                    num_positives = label_counts[label] + 0.5
                    num_negatives = total_label_count - num_positives + 0.5
                    self._pos_weight[label_index] = num_negatives / num_positives
            else:
                torch.nn.init.constant_(self._pos_weight, 1.0)
                for label, weight in self._class_weights.items():  # type: ignore[union-attr]
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
                F.binary_cross_entropy_with_logits(
                    logits,
                    labels.bool().float(),
                    pos_weight=self._pos_weight,
                    reduction="sum",
                )
                / logits.size(0),
            )

        return output
