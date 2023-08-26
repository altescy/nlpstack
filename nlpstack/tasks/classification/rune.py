import tempfile
import warnings
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Literal, Mapping, Optional, Sequence, Set, Union

import numpy

from nlpstack.data import BasicBatchSampler, DataLoader, Vocabulary
from nlpstack.data.indexers import SingleIdTokenIndexer, TokenIndexer
from nlpstack.data.tokenizers import Tokenizer, WhitespaceTokenizer
from nlpstack.data.util import batched
from nlpstack.evaluation import MultiMetrics
from nlpstack.rune import RuneForTorch
from nlpstack.torch.modules.seq2vec_encoders import BagOfEmbeddings
from nlpstack.torch.modules.text_embedders import TextEmbedder
from nlpstack.torch.modules.token_embedders import Embedding
from nlpstack.torch.training import TorchTrainer
from nlpstack.torch.training.callbacks import Callback
from nlpstack.torch.training.optimizers import AdamFactory

from .datamodules import BasicClassificationDataModule
from .metrics import Accuracy, ClassificationMetric
from .torch import TorchBasicClassifier
from .types import ClassificationExample, ClassificationInference, ClassificationPrediction

logger = getLogger(__name__)


class BasicClassifier(
    RuneForTorch[
        ClassificationExample,
        ClassificationInference,
        ClassificationPrediction,
    ]
):
    """
    A basic neural text classification model.

    Args:
        min_df: The minimum document frequency of the tokens. If `float`, the minimum
            document frequency is the fraction of the total number of documents. Defaults to `1`.
        max_df: The maximum document frequency of the tokens. If `float`, the maximum
            document frequency is the fraction of the total number of documents. Defaults to `1.0`.
        pad_token: The padding token. You can specify a different padding token for each
            namespace by passing a mapping from namespace to padding token. Defaults to `"@@PADDING@@"`.
        oov_token: The out-of-vocabulary (OOV) token. You can specify a different OOV token
            for each namespace by passing a mapping from namespace to OOV token. Defaults to `"@@UNKNOWN@@"`.
        vocab: The vocabulary. If given, the vocabulary-related arguments will be ignored, otherwise
            the vocabulary will be constructed from the data. Defaults to `None`.
        tokenizer: The tokenizer.
        token_indexers: The token indexers to index the tokens.
        datamodule: The data module. If given, the data module related arguments will be ignored,
        dropout: The dropout rate. Defaults to `None`.
        class_weights: The class weights. If `"balanced"`, the class weights will be set to
            the inverse of the class frequencies. You can specify a different class weight for
            each class by passing a mapping from class to class weight. Defaults to `None`.
        classifier: The classifier. If given, the model related arguments will be ignored.
        max_epochs: The maximum number of epochs. Defaults to `4`.
        batch_size: The batch size. Defaults to `32`.
        learning_rate: The learning rate. Defaults to `1e-3`.
        training_callbacks: The training callbacks for `TorchTrainer`. Defaults to `None`.
        trainer: The trainer for training the model. If given, the trainer related arguments will be ignored,
            otherwise the trainer will be constructed from the related arguments. Defaults to `None`.
        metric: The metric for evaluation. Defaults to `Accuracy()`.
        random_seed: The random seed. Defaults to `None`.
    """

    def __init__(
        self,
        *,
        # data configuration
        min_df: Union[int, float, Mapping[str, Union[int, float]]] = 1,
        max_df: Union[int, float, Mapping[str, Union[int, float]]] = 1.0,
        pad_token: Union[str, Mapping[str, str]] = "@@PADDING@@",
        oov_token: Union[str, Mapping[str, str]] = "@@UNKNOWN@@",
        vocab: Optional[Vocabulary] = None,
        tokenizer: Optional[Tokenizer] = None,
        token_indexers: Optional[Mapping[str, TokenIndexer]] = None,
        datamodule: Optional[BasicClassificationDataModule] = None,
        # model configuration
        dropout: Optional[float] = None,
        class_weights: Optional[Union[Literal["balanced"], Mapping[str, float]]] = None,
        classifier: Optional[TorchBasicClassifier] = None,
        # training configuration
        max_epochs: int = 4,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        training_callbacks: Optional[Sequence[Callback]] = None,
        trainer: Optional[TorchTrainer] = None,
        # evaluation configuration
        metric: Optional[Union[ClassificationMetric, Sequence[ClassificationMetric]]] = None,
        # other configuration
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if datamodule is None:
            if vocab is None:
                default_token_namespace = "tokens"
                min_df = {default_token_namespace: min_df} if isinstance(min_df, (int, float)) else min_df
                max_df = {default_token_namespace: max_df} if isinstance(max_df, (int, float)) else max_df
                pad_token = {default_token_namespace: pad_token} if isinstance(pad_token, str) else pad_token
                oov_token = {default_token_namespace: oov_token} if isinstance(oov_token, str) else oov_token
                special_tokens: Dict[str, Set[str]] = {}
                for namespace, token in pad_token.items():
                    special_tokens.setdefault(namespace, set()).add(token)
                for namespace, token in oov_token.items():
                    special_tokens.setdefault(namespace, set()).add(token)
                vocab = Vocabulary(
                    min_df=min_df,
                    max_df=max_df,
                    pad_token=pad_token,
                    oov_token=oov_token,
                    special_tokens=special_tokens,
                )
            else:
                if (min_df, max_df, pad_token, oov_token) != (1, 1.0, "@@PADDING@@", "@@UNKNOWN@@"):
                    warnings.warn(
                        "Ignoring min_df, max_df, pad_token, and oov_token because vocab is given.",
                        UserWarning,
                    )

            if tokenizer is None:
                tokenizer = WhitespaceTokenizer()

            if token_indexers is None:
                token_indexers = {"tokens": SingleIdTokenIndexer()}

            datamodule = BasicClassificationDataModule(
                vocab=vocab,
                tokenizer=tokenizer,
                token_indexers=token_indexers,
            )

        if classifier is None:
            classifier = TorchBasicClassifier(
                embedder=TextEmbedder({"tokens": Embedding(64)}),
                encoder=BagOfEmbeddings(64),
                dropout=dropout,
                class_weights=class_weights,
            )
        else:
            if (dropout, class_weights) != (None, None):
                warnings.warn(
                    "Ignoring dropout and class_weights because classifier is given.",
                    UserWarning,
                )

        if trainer is None:
            trainer = TorchTrainer(
                train_dataloader=DataLoader(BasicBatchSampler(batch_size=batch_size, shuffle=True)),
                valid_dataloader=DataLoader(BasicBatchSampler(batch_size=batch_size, shuffle=False)),
                max_epochs=max_epochs,
                optimizer_factory=AdamFactory(lr=learning_rate),
                callbacks=training_callbacks,
            )
        else:
            if (max_epochs, batch_size, learning_rate, training_callbacks) != (4, 32, 1e-3, None):
                warnings.warn(
                    "Ignoring max_epochs, batch_size, learning_rate, and training_callbacks because trainer is given.",
                    UserWarning,
                )

        if metric is None:
            metric = Accuracy()

        super().__init__(
            datamodule=datamodule,
            model=classifier,
            trainer=trainer,
            metric=metric,
            random_seed=random_seed,
            **kwargs,
        )


class FastTextClassifier(
    RuneForTorch[
        ClassificationExample,
        ClassificationInference,
        ClassificationPrediction,
    ]
):
    """
    A fastText classifier.

    Each parameter corresponds to the original fastText implementation. For more details, please refer to here:
        https://github.com/facebookresearch/fastText/tree/main/python#train_supervised-parameters

    Args:
        tokenizer: The tokenizer for tokenizing the text. Defaults to `WhitespaceTokenizer()`.
        learning_rate: The learning rate. Defaults to `0.1`.
        dim: The dimension of word vectors. Defaults to `100`.
        ws: The size of the context window. Defaults to `5`.
        epoch: The number of epochs. Defaults to `5`.
        min_count: The minimum number of word occurrences. Defaults to `1`.
        min_count_label: The minimum number of label occurrences. Defaults to `1`.
        minn: The min length of char ngram. Defaults to `0`.
        maxn: The max length of char ngram. Defaults to `0`.
        neg: The number of negatives sampled. Defaults to `5`.
        word_ngrams: The max length of word ngram. Defaults to `1`.
        loss: The loss function. Defaults to `"softmax"`.
        bucket: The number of buckets. Defaults to `2000000`.
        thread: The number of threads. Defaults to number of CPUs.
        lr_update_rate: The rate of updates for the learning rate. Defaults to `100`.
        t: The sampling threshold. Defaults to `0.0001`.
        pretrained_vectors: The path to pretrained vectors. Defaults to `None`.
        seed: The random seed. Defaults to `None`.
        autotune_duration: The duration of autotune in seconds. If given, the model will be
            trained with autotune. Defaults to `None`.
        metric: The metric for evaluation. Defaults to `Accuracy()`.
    """

    def __init__(
        self,
        *,
        tokenizer: Optional[Tokenizer] = None,
        lr: Optional[float] = None,
        dim: Optional[int] = None,
        ws: Optional[int] = None,
        epoch: Optional[int] = None,
        min_count: Optional[int] = None,
        min_count_label: Optional[int] = None,
        minn: Optional[int] = None,
        maxn: Optional[int] = None,
        neg: Optional[int] = None,
        word_ngrams: Optional[int] = None,
        loss: Optional[Literal["softmax", "ns", "hs", "ova"]] = None,
        bucket: Optional[int] = None,
        thread: Optional[int] = None,
        lr_update_rate: Optional[int] = None,
        t: Optional[float] = None,
        pretrained_vectors: Optional[str] = None,
        seed: Optional[int] = None,
        autotune_duration: Optional[int] = None,
        metric: Optional[Union[ClassificationMetric, Sequence[ClassificationMetric]]] = None,
    ) -> None:
        import fasttext

        self._tokenizer = tokenizer or WhitespaceTokenizer()
        self._training_config = self._construct_training_config(
            lr=lr,
            dim=dim,
            ws=ws,
            epoch=epoch,
            min_count=min_count,
            min_count_label=min_count_label,
            minn=minn,
            maxn=maxn,
            neg=neg,
            word_ngrams=word_ngrams,
            loss=loss,
            bucket=bucket,
            thread=thread,
            lr_update_rate=lr_update_rate,
            t=t,
            pretrained_vectors=pretrained_vectors,
            seed=seed,
            autotune_duration=autotune_duration,
        )

        self._model: Optional[fasttext.FastText] = None
        self._labels: Set[str] = set()

        self.metric: ClassificationMetric
        if metric is None:
            self.metric = Accuracy()
        elif isinstance(metric, Sequence):
            self.metric = MultiMetrics(metric)  # type: ignore[assignment]
        else:
            self.metric = metric

    @staticmethod
    def _construct_training_config(**kwargs: Any) -> Mapping[str, Any]:
        return {
            "".join(x if i == 0 else x.capitalize() for i, x in enumerate(key.split("_"))): value
            for key, value in kwargs.items()
            if value is not None
        }

    def train(
        self,
        train_dataset: Sequence[ClassificationExample],
        valid_dataset: Optional[Sequence[ClassificationExample]] = None,
        resources: Optional[Mapping[str, Any]] = None,
        **kwargs: Any,
    ) -> "FastTextClassifier":
        import fasttext

        self._model = None
        self._labels = set()

        with tempfile.TemporaryDirectory() as _workdir:
            workdir = Path(_workdir)

            train_filename = workdir / "train.txt"
            with train_filename.open("w") as f:
                for example in train_dataset:
                    assert example.label is not None
                    label = example.label
                    tokens = self._tokenizer.tokenize(example.text) if isinstance(example.text, str) else example.text
                    text = " ".join(token.surface for token in tokens)
                    f.write(f"__label__{label} {text}\n")
                    self._labels.add(label)

            valid_filename: Optional[Path] = None
            if valid_dataset is not None:
                valid_filename = workdir / "valid.txt"
                with valid_filename.open("w") as f:
                    for example in valid_dataset:
                        assert example.label is not None
                        label = example.label
                        tokens = (
                            self._tokenizer.tokenize(example.text) if isinstance(example.text, str) else example.text
                        )
                        text = " ".join(token.surface for token in tokens)
                        f.write(f"__label__{label} {text}\n")

            training_config = dict(self._training_config)
            if "autotuneDuration" in training_config:
                if valid_filename is None:
                    raise ValueError("valid_dataset is requred for autotune")
                training_config["autotuneValidationFile"] = str(valid_filename)

            self._model = fasttext.train_supervised(
                input=str(train_filename),
                **training_config,
            )

        return self

    def predict(  # type: ignore[override]
        self,
        dataset: Iterable[ClassificationExample],
        *,
        k: Optional[int] = None,
        threshold: float = 0.0,
    ) -> Iterator[ClassificationPrediction]:
        if self._model is None:
            raise RuntimeError("model is not trained")
        k = k or len(self._labels)
        for example in dataset:
            tokens = self._tokenizer.tokenize(example.text) if isinstance(example.text, str) else example.text
            text = " ".join(token.surface for token in tokens)
            labels, probs = self._model.predict(text, k=k, threshold=threshold)
            yield ClassificationPrediction(
                top_probs=probs.tolist(),
                top_labels=[label.replace("__label__", "") for label in labels],
                metadata=example.metadata,
            )

    def evaluate(  # type: ignore[override]
        self,
        dataset: Iterable[ClassificationExample],
        *,
        batch_size: int = 32,
    ) -> Mapping[str, float]:
        if self._model is None:
            raise RuntimeError("model is not trained")
        label_to_index = {label: i for i, label in enumerate(self._labels)}
        self.metric.reset()
        for batch in batched(dataset, batch_size):
            assert all(example.label is not None for example in batch)
            texts = [
                " ".join(
                    token.surface
                    for token in (
                        self._tokenizer.tokenize(example.text) if isinstance(example.text, str) else example.text
                    )
                )
                for example in batch
            ]
            labels = numpy.array([label_to_index[example.label] for example in batch if example.label], dtype=int)
            predicted_labels, probs = self._model.predict(texts, k=len(self._labels))
            label_indices = numpy.array(
                [[label_to_index[label.replace("__label__", "")] for label in labels] for labels in predicted_labels]
            ).argsort(axis=1)
            probs = numpy.take_along_axis(numpy.array(probs), label_indices, axis=1)
            inference = ClassificationInference(probs=probs, labels=labels)
            self.metric.update(inference)
        return self.metric.compute()

    def __getstate__(self) -> Dict[str, Any]:
        state = dict(self.__dict__)
        if state["_model"] is not None:
            model = state.pop("_model")
            with tempfile.TemporaryDirectory() as _workdir:
                workdir = Path(_workdir)
                model_filename = workdir / "model.bin"
                model.save_model(str(model_filename))
                state["_model_binary"] = model_filename.read_bytes()
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        import fasttext

        if "_model_binary" in state:
            model_binary = state.pop("_model_binary")
            with tempfile.TemporaryDirectory() as _workdir:
                workdir = Path(_workdir)
                model_filename = workdir / "model.bin"
                model_filename.write_bytes(model_binary)
                state["_model"] = fasttext.load_model(str(model_filename))
        self.__dict__.update(state)
