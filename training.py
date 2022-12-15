import datetime
import sqlite3
import timeit
import traceback
from collections import namedtuple
from typing import get_args
from unittest import mock

import torch
from torch.multiprocessing import Process

from data import ActFn, Task, TrainConfig, sparsities
from toy_model import train_model
from training_db import insert_train_result

act_fns = get_args(ActFn)  #  ["ReLU", "GeLU", "SoLU"]
DUPS = 2


def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


SizeConfig = namedtuple("SizeConfig", ["neurons", "features"])

size_configs = [
    SizeConfig(5, 20),  # 5 neurons, 20 features, already trained
    # SizeConfig(40, 100),
]

tasks = get_args(Task)


@mock.patch("tqdm.auto.tqdm", notqdm)
def train_one(act_fn, use_ln, s, size_config, task):
    try:
        start = timeit.default_timer()
        config = TrainConfig(
            "LayerNormToyModel" if use_ln else "ToyModel",
            s=s,
            i=0.8,
            task=task,
            steps=(100_000 if use_ln else 50_000),
            act_fn=act_fn,
            args=dict(
                neurons=size_config.neurons,
                features=size_config.features,
            ),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
        print(f"{timestamp}: training {act_fn}, {use_ln}, {s}")
        train_result = train_model(config, device)
        stop = timeit.default_timer()
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
        print(f"{timestamp}: inserting {act_fn}, {use_ln}, {s} (after {stop - start}s)")
        con = sqlite3.connect("training.db")
        cur = con.cursor()
        insert_train_result(train_result, None, cur, con)
    except Exception:
        print(traceback.format_exc())
        raise


def train_some(config_batch):
    for config in config_batch:
        train_one(*config)


if __name__ == "__main__":
    configurations = [
        (act_fn, False, s, SizeConfig(5, 20), task)
        for act_fn in act_fns
        # for use_ln in (False, True)
        for s in sparsities
        # for size_config in size_configs
        for task in ["ID", "SQUARE", "MAX", "MIN"]
    ] * DUPS

    num_processes = 16
    processes = []

    work = [[]] * num_processes
    for i in range(len(configurations)):
        work[i % num_processes].append(configurations[i])

    for config_batch in work:
        p = Process(target=train_some, args=(config_batch,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
