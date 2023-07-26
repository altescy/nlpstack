import dataclasses
from logging import getLogger
from typing import Any, Literal, Mapping, Optional, Union, cast

import torch
import torch.nn.functional as F

from nlpstack.torch.model import TorchModel
from nlpstack.torch.modules.count_embedder import CountEmbedder
from nlpstack.torch.modules.lazy import LazyLinearOutput

from .datamodules import TopicModelingDataModule
from .types import TopicModelingInference

logger = getLogger(__name__)


@dataclasses.dataclass
class TorchProdLDAOutput:
    inference: TopicModelingInference
    loss: Optional[torch.FloatTensor] = None


class TorchProdLDA(TorchModel[TopicModelingInference]):
    class HiddenToLogNormal(torch.nn.Module):
        def __init__(self, hidden_size: int, num_topics: int) -> None:
            super().__init__()
            self.fcmu = torch.nn.Linear(hidden_size, num_topics)
            self.fclv = torch.nn.Linear(hidden_size, num_topics)
            self.bnmu = torch.nn.BatchNorm1d(num_topics)
            self.bnlv = torch.nn.BatchNorm1d(num_topics)

        def forward(self, hidden: torch.Tensor) -> torch.distributions.LogNormal:
            mu = self.bnmu(self.fcmu(hidden))
            lv = self.bnlv(self.fclv(hidden))
            return torch.distributions.LogNormal(mu, (0.5 * lv).exp())  # type: ignore[no-untyped-call]

    class HiddenToDirichlet(torch.nn.Module):
        def __init__(self, hidden_size: int, num_topics: int) -> None:
            super().__init__()
            self.fc = torch.nn.Linear(hidden_size, num_topics)
            self.bn = torch.nn.BatchNorm1d(num_topics)

        def forward(self, hidden: torch.Tensor) -> torch.distributions.Dirichlet:
            alphas = self.bn(self.fc(hidden)).exp().cpu()
            return torch.distributions.Dirichlet(alphas)  # type: ignore[no-untyped-call]

    def __init__(
        self,
        num_topics: int,
        hidden_dim: int,
        dropout: Optional[float] = 0.0,
        prior: Literal["dirichlet", "lognormal"] = "dirichlet",
        token_namespace: str = "tokens",
    ) -> None:
        dropout = dropout or 0.0
        if prior not in ("dirichlet", "lognormal"):
            raise ValueError(f"prior must be one of 'dirichlet' or 'lognormal', got {prior}")

        super().__init__()

        self._num_topics = num_topics
        self._embedder = CountEmbedder()
        self._encoder = torch.nn.Sequential(
            torch.nn.LazyLinear(hidden_dim),
            torch.nn.Softplus(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.Softplus(),
            torch.nn.Dropout(dropout),
        )
        self._decoder = LazyLinearOutput(num_topics)
        self._projector = (
            TorchProdLDA.HiddenToDirichlet(hidden_dim, num_topics)
            if prior == "dirichlet"
            else TorchProdLDA.HiddenToLogNormal(hidden_dim, num_topics)
        )
        self._batchnorm = torch.nn.LazyBatchNorm1d()
        self._dropout = torch.nn.Dropout(dropout)
        self._token_namespace = token_namespace

    def standard_prior_like(
        self, posterior: Union[torch.distributions.Dirichlet, torch.distributions.LogNormal]
    ) -> Union[torch.distributions.Dirichlet, torch.distributions.LogNormal]:
        if isinstance(posterior, torch.distributions.LogNormal):
            loc = torch.zeros_like(posterior.loc)
            scale = torch.ones_like(posterior.scale)
            return torch.distributions.LogNormal(loc, scale)  # type: ignore[no-untyped-call]
        if isinstance(posterior, torch.distributions.Dirichlet):
            alphas = torch.ones_like(posterior.concentration)
            return torch.distributions.Dirichlet(alphas)  # type: ignore[no-untyped-call]
        raise ValueError(f"Unknown posterior type {type(posterior)}")

    def setup(
        self,
        *args: Any,
        datamodule: TopicModelingDataModule,
        **kwargs: Any,
    ) -> None:
        vocab = datamodule.vocab
        super().setup(*args, datamodule=datamodule, vocab=vocab, **kwargs)
        self._decoder.initialize_parameters(out_features=vocab.get_vocab_size(self._token_namespace))

    def forward(  # type: ignore[override]
        self,
        text: Mapping[str, Mapping[str, torch.Tensor]],
        *args: Any,
    ) -> TorchProdLDAOutput:
        embedding = self._embedder(text)
        hidden = self._encoder(embedding)
        posterior = self._projector(hidden)

        if self.training:
            topic_distribution = posterior.rsample().to(embedding.device)
        else:
            topic_distribution = posterior.mean.to(embedding.device)
        topic_distribution = topic_distribution / topic_distribution.sum(dim=1, keepdim=True)

        output = F.log_softmax(self._batchnorm(self._decoder(topic_distribution)))

        prior = self.standard_prior_like(posterior)
        nll = -torch.sum(embedding * output)
        kld = torch.distributions.kl_divergence(posterior, prior).sum().to(embedding.device)
        ppl = torch.exp(nll / embedding.sum())

        loss = cast(torch.FloatTensor, (nll + kld) / embedding.size(0))
        inference = TopicModelingInference(
            topic_distribution=topic_distribution.detach().cpu().numpy(),
            token_counts=embedding.detach().cpu().numpy(),
            perplexity=float(ppl.detach().cpu().item()),
        )

        return TorchProdLDAOutput(inference=inference, loss=loss)

    def get_topics(self) -> torch.FloatTensor:
        """Get the term-topic matrix.

        Returns:
            torch.FloatTensor: The term-topic matrix of shape (num_topics, vocab_size). Each row
                is an unnormalized distribution over the vocabulary.
        """
        return cast(torch.FloatTensor, self._decoder.weight.detach().clone().cpu().T)
