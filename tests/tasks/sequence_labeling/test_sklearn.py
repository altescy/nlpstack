from nlpstack.tasks.sequence_labeling.metrics import SpanBasedF1, TokenBasedAccuracy
from nlpstack.tasks.sequence_labeling.sklearn import SklearnBasicSequenceLabeler
from nlpstack.tasks.sequence_labeling.torch import TorchSequenceLabeler
from nlpstack.torch.modules.crf import CrfDecoder
from nlpstack.torch.modules.seq2seq_encoders import LstmSeq2SeqEncoder
from nlpstack.torch.modules.text_embedders import TextEmbedder
from nlpstack.torch.modules.token_embedders import Embedding


def test_basic_sequence_labeler() -> None:
    X = [
        ["John", "Smith", "was", "born", "in", "New", "York", "."],
        ["New", "York", "is", "a", "city", "in", "the", "United", "States", "."],
    ]
    y = [
        ["B-PER", "I-PER", "O", "O", "O", "B-LOC", "I-LOC", "O"],
        ["B-LOC", "I-LOC", "O", "O", "O", "O", "O", "B-LOC", "I-LOC", "O"],
    ]

    sequence_labeler = SklearnBasicSequenceLabeler(
        max_epochs=30,
        learning_rate=1e-2,
        sequence_labeler=TorchSequenceLabeler(
            embedder=TextEmbedder({"tokens": Embedding(32)}),
            encoder=LstmSeq2SeqEncoder(32, 16, 1, bidirectional=True),
            decoder=CrfDecoder("BIO"),
        ),
        metric=[SpanBasedF1(), TokenBasedAccuracy()],
        random_seed=42,
    )
    sequence_labeler.fit(X, y)

    predictions = sequence_labeler.predict(X)
    assert predictions == y

    score = sequence_labeler.score(X, y)
    assert score == 1.0

    metrics = sequence_labeler.compute_metrics(X, y)
    assert set(metrics.keys()) == {
        "token_accuracy",
        "precision_overall",
        "recall_overall",
        "f1_overall",
        "precision_PER",
        "recall_PER",
        "f1_PER",
        "precision_LOC",
        "recall_LOC",
        "f1_LOC",
    }
    assert all(abs(value - 1.0) < 1e-5 for value in metrics.values()), metrics
