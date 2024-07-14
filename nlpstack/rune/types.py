from typing import Literal, TypeVar

Example = TypeVar("Example")
Prediction = TypeVar("Prediction")
SetupParams = TypeVar("SetupParams")
PredictionParams = TypeVar("PredictionParams")
EvaluationParams = TypeVar("EvaluationParams")
SetupMode = Literal["training", "prediction", "evaluation"]
