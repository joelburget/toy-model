import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Literal, List
import pickle
import os


Task = Literal["ID", "SQUARE", "ABS"]
ModelName = Literal[
    "ToyModel",
    "ReluHiddenLayerModel",
    "ReluHiddenLayerModelVariation",
    "MultipleHiddenLayerModel",
    "MlpModel",
]
sparsities = [0, 0.7, 0.9, 0.97, 0.99, 0.997, 0.999]


@dataclass
class TrainConfig:
    model_name: ModelName
    s: float  # sparsity
    i: float = 0.7  # importance base
    points: int = 8096
    steps: int = 40_000
    task: Task = "ID"


@dataclass
class TrainResult:
    model: nn.Module
    config: TrainConfig
    losses: List[float]
    train: torch.Tensor
    test: torch.Tensor

    def save(self, path, mkdir=True):
        if mkdir:
            os.mkdir(path)
        for name, attribute in self.__dict__.items():
            name = ".".join((name, "pkl"))
            with open("/".join((path, name)), "wb") as f:
                pickle.dump(attribute, f)

    @classmethod
    def load(cls, path):
        my_model = {}
        for name in cls.__annotations__:
            file_name = ".".join((name, "pkl"))
            with open("/".join((path, file_name)), "rb") as f:
                my_model[name] = pickle.load(f)
        return cls(**my_model)
