import dataclasses
from typing import Literal, Optional

from optuna.pruners import BasePruner
from optuna.samplers import BaseSampler

from nlpstack.common import FromJsonnet


@dataclasses.dataclass
class OptunaConfig(FromJsonnet):
    metric: str
    direction: Literal["minimize", "maximize"] = "minimize"
    sampler: Optional[BaseSampler] = None
    pruner: Optional[BasePruner] = None
