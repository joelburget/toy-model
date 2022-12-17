import os
import pickle
from dataclasses import dataclass, field
from typing import List, Literal, Tuple

import torch
import torch.nn as nn

Task = Literal["ID", "SQUARE", "ABS", "MAX", "MIN"]
ModelName = Literal[
    "ToyModel",
    "ReluHiddenLayerModel",
    "HiddenLayerModel",
    "ReluHiddenLayerModelVariation",
    "HiddenLayerModelVariation",
    # "MultipleHiddenLayerModel",
    "MlpModel",
    "ResidualModel",
]
sparsities = [0, 0.7, 0.9, 0.97, 0.99, 0.997, 0.999]
ActFn = Literal["ReLU", "GeLU", "SoLU"]


@dataclass
class TrainConfig:
    model_name: ModelName
    s: float  # sparsity
    i: float = 0.7  # importance base
    points: int = 8096
    steps: int = 40_000
    task: Task = "ID"
    regularization_coeff: float = 0.0  # for L1 regularization
    act_fn: ActFn = "ReLU"
    args: dict = field(default_factory=dict)


@dataclass
class TrainResult:
    model: nn.Module
    config: TrainConfig
    losses: List[float]
    train: torch.Tensor
    test: torch.Tensor
    checkpoints: List[Tuple[int, dict]] = field(default_factory=list)

    def save(self, path, mkdir=True):
        if mkdir:
            try:
                os.mkdir(path)
            except FileExistsError:
                pass
        for name, attribute in self.__dict__.items():
            name = ".".join((name, "pkl"))
            with open("/".join((path, name)), "wb") as f:
                pickle.dump(attribute, f)

    @classmethod
    def load(cls, path, load_checkpoints=False):
        my_model = {}
        annotations = cls.__annotations__.copy()
        del annotations["checkpoints"]

        for name in annotations:
            file_name = ".".join((name, "pkl"))
            with open("/".join((path, file_name)), "rb") as f:
                my_model[name] = pickle.load(f)

        if load_checkpoints:
            try:
                with open("/".join((path, "checkpoints.pkl")), "rb") as f:
                    my_model["checkpoints"] = pickle.load(f)
            except FileNotFoundError:
                my_model["checkpoints"] = []
        else:
            my_model["checkpoints"] = []

        return cls(**my_model)
