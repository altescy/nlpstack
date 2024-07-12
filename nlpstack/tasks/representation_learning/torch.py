import dataclasses
from logging import getLogger
from typing import Any, Mapping, Optional, Sequence, cast

import torch
import torch.nn.functional as F

from nlpstack.integrations.torch.model import TorchModel
from nlpstack.integrations.torch.modules.feedforward import FeedForward
from nlpstack.integrations.torch.modules.seq2seq_encoders import Seq2SeqEncoder
from nlpstack.integrations.torch.modules.seq2vec_encoders import Seq2VecEncoder
from nlpstack.integrations.torch.modules.text_embedders import TextEmbedder
from nlpstack.integrations.torch.util import get_mask_from_text

from .datamodules import RepresentationLearningDataModule
from .types import RepresentationLearningInference

logger = getLogger(__name__)


@dataclasses.dataclass
class TorchUnsupervisedSimCSEOutput:
    inference: RepresentationLearningInference
    loss: Optional[torch.FloatTensor] = None


class TorchUnsupervisedSimCSE(TorchModel[RepresentationLearningInference]):
    """
    An unsupervised SimCSE model for PyTorch.

    Args:
        embedder: The text embedder.
        encoder: The sequence-to-vector encoder for building the text representation.
        contextualizer: The sequence-to-sequence encoder for contextualizing the text representation.
            This is applied before the encoder. If `None`, no contextualizer is applied. Defaults to
            `None`.
        feedforward: The feedforward layer applied to the text representation. This is applied after
            the encoder. If `None`, no feedforward layer is applied. Defaults to `None`.
        dropout: The dropout rate. Defaults to `None`.
        temperature: The tempareture parameter for SimCSE training. Defaults to `0.05`.
        use_feedforward_for_only_training: If `True`, the feedforward is used only for trainig. Defaults to `False`.
    """

    def __init__(
        self,
        embedder: TextEmbedder,
        encoder: Seq2VecEncoder,
        contextualizer: Optional[Seq2SeqEncoder] = None,
        feedforward: Optional[FeedForward] = None,
        dropout: Optional[float] = None,
        temperature: Optional[float] = None,
        use_feedforward_for_only_training: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self._embedder = embedder
        self._encoder = encoder
        self._contextualizer = contextualizer
        self._feedforward = feedforward
        self._dropout = torch.nn.Dropout(dropout) if dropout is not None else None
        self._temperature = temperature or 0.05
        self._use_feedforward_for_only_training = use_feedforward_for_only_training or False

    def setup(self, *args: Any, datamodule: RepresentationLearningDataModule, **kwargs: Any) -> None:
        super().setup(*args, datamodel=datamodule, vocab=datamodule.vocab, **kwargs)

    def forward(  # type: ignore[override]
        self,
        text: Mapping[str, Mapping[str, torch.Tensor]],
        metadata: Optional[Sequence[Mapping[str, Any]]] = None,
        *args: Any,
    ) -> TorchUnsupervisedSimCSEOutput:
        def compute_embeddings(text: Mapping[str, Mapping[str, torch.Tensor]]) -> torch.FloatTensor:
            mask = get_mask_from_text(text)
            embedding = self._embedder(text)
            if self._contextualizer is not None:
                embedding = self._contextualizer(embedding, mask)
            embeddings = self._encoder(embedding, mask)
            if self._dropout is not None:
                embedding = self._dropout(embedding)
            if self._feedforward is not None:
                if not self._use_feedforward_for_only_training or self.training:
                    embeddings = self._feedforward(embeddings)
            return cast(torch.FloatTensor, embeddings)

        embeddings = compute_embeddings(text)
        inference = RepresentationLearningInference(embeddings=embeddings.detach().cpu().numpy(), metadata=metadata)
        output = TorchUnsupervisedSimCSEOutput(inference=inference)

        if self.training:
            embeddings_for_simcse = compute_embeddings(text)
            similarity_matrix = F.cosine_similarity(embeddings.unsqueeze(1), embeddings_for_simcse.unsqueeze(0), dim=-1)
            similarity_matrix = similarity_matrix / self._temperature
            labels = torch.arange(embeddings.size(0), dtype=torch.long, device=embeddings.device)
            output.loss = cast(torch.FloatTensor, F.cross_entropy(similarity_matrix, labels))

        return output
