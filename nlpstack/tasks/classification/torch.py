import dataclasses
from typing import Any, Literal, Mapping, NamedTuple, Optional, Sequence, Union, cast

import torch
import torch.nn.functional as F

from nlpstack.integrations.torch.model import TorchModel
from nlpstack.integrations.torch.modules.feedforward import FeedForward
from nlpstack.integrations.torch.modules.heads import ClassificationHead, Head
from nlpstack.integrations.torch.modules.seq2seq_encoders import Seq2SeqEncoder
from nlpstack.integrations.torch.modules.seq2vec_encoders import Seq2VecEncoder
from nlpstack.integrations.torch.modules.text_embedders import TextEmbedder
from nlpstack.integrations.torch.util import get_mask_from_text

from .datamodules import BasicClassificationDataModule
from .types import ClassificationInference


@dataclasses.dataclass
class BasicClassifierOutput:
    inference: ClassificationInference
    logits: torch.FloatTensor
    loss: Optional[torch.FloatTensor] = None


class TorchBasicClassifier(
    TorchModel[
        ClassificationInference,
        "TorchBasicClassifier.Inputs",
        "TorchBasicClassifier.Params",
    ]
):
    """
    A basic classifier for PyTorch.

    Args:
        embedder: The text embedder.
        encoder: The sequence-to-vector encoder for building the text representation.
        contextualizer: The sequence-to-sequence encoder for contextualizing the text representation.
            This is applied before the encoder. If `None`, no contextualizer is applied. Defaults to
            `None`.
        feedforward: The feedforward layer applied to the text representation. This is applied after
            the encoder. If `None`, no feedforward layer is applied. Defaults to `None`.
        head: The head for the classifier. If not given, `ClassificationHead` is used. Defaults to `None`.
        dropout: The dropout rate. Defaults to `None`.
        class_weights: The class weights. If `None`, no class weights are used. If `"balanced"`, the
            class weights are set to be inversely proportional to the class frequencies. Otherwise,
            the class weights are set to the given mapping. Defaults to `None`.
        threshold: The threshold for the prediction. If `None`, all the classes are returned as the
            prediction. Otherwise, only the classes with the probabilities greater than the threshold
            are returned. Defaults to `None`.
        label_namespace: The namespace of the labels. Defaults to `"labels"`.
    """

    class Inputs(NamedTuple):
        text: Mapping[str, Mapping[str, torch.Tensor]]
        label: Optional[torch.LongTensor] = None
        metadata: Optional[Sequence[Any]] = None

    class Params(NamedTuple):
        top_k: Optional[int] = None
        threshold: Optional[float] = None

    def __init__(
        self,
        embedder: TextEmbedder,
        encoder: Seq2VecEncoder,
        contextualizer: Optional[Seq2SeqEncoder] = None,
        feedforward: Optional[FeedForward] = None,
        head: Optional[Head] = None,
        dropout: Optional[float] = None,
        class_weights: Optional[Union[Literal["balanced"], Mapping[str, float]]] = None,
        threshold: Optional[float] = None,
        label_namespace: str = "labels",
    ) -> None:
        super().__init__()
        self._embedder = embedder
        self._encoder = encoder
        self._contextualizer = contextualizer
        self._feedforward = feedforward
        self._head = head or ClassificationHead(
            input_dim=(feedforward or encoder).get_output_dim(),
            namespace=label_namespace,
        )
        self._dropout = torch.nn.Dropout(dropout) if dropout is not None else None

        self._class_weights = class_weights
        self._loss_weight = None if class_weights is None else torch.nn.UninitializedParameter(requires_grad=False)

        self._loss = torch.nn.CrossEntropyLoss()

        self._threshold = threshold
        self._label_namespace = label_namespace

    def setup(
        self,
        *args: Any,
        datamodule: BasicClassificationDataModule,
        **kwargs: Any,
    ) -> None:
        super().setup(*args, datamodule=datamodule, vocab=datamodule.vocab, **kwargs)
        vocab = datamodule.vocab
        num_labels = vocab.get_vocab_size(self._label_namespace)
        if self._class_weights is not None and self._loss_weight is not None:
            self._loss_weight.materialize((num_labels,))
            if self._class_weights == "balanced":
                label_counts = {
                    key: value + 1  # plus one for smoothing
                    for key, value in vocab.get_token_to_count(self._label_namespace).items()
                }
                total_label_count = sum(label_counts.values())
                for label_index, label in vocab.get_index_to_token(self._label_namespace).items():
                    self._loss_weight[label_index] = total_label_count / label_counts[label]
            else:
                torch.nn.init.constant(self._loss_weight, 1.0)
                for label, weight in self._class_weights.items():  # type: ignore[union-attr]
                    label_index = vocab.get_index_by_token(self._label_namespace, label)
                    self._loss_weight[label_index] = weight

    def forward(
        self,
        inputs: "TorchBasicClassifier.Inputs",
        params: Optional["TorchBasicClassifier.Params"] = None,
    ) -> BasicClassifierOutput:
        text, label, metadata = inputs
        top_k, threshold = params or TorchBasicClassifier.Params()

        mask = get_mask_from_text(text)

        embeddings = self._embedder(text)

        if self._contextualizer is not None:
            embeddings = self._contextualizer(embeddings, mask=mask)

        encodings = self._encoder(embeddings, mask=mask)

        if self._feedforward is not None:
            encodings = self._feedforward(encodings)

        if self._dropout is not None:
            encodings = self._dropout(encodings)

        logits = cast(torch.FloatTensor, self._head(encodings))
        probs = cast(torch.FloatTensor, torch.nn.functional.softmax(logits, dim=-1))

        inference = ClassificationInference(
            probs=probs.detach().cpu().numpy(),
            metadata=metadata,
            top_k=top_k,
            threshold=self._threshold if threshold is None else threshold,
        )
        output = BasicClassifierOutput(inference=inference, logits=logits)

        if label is not None:
            inference.labels = label.detach().cpu().numpy()
            output.loss = cast(torch.FloatTensor, F.cross_entropy(logits, label.long(), weight=self._loss_weight))

        return output
