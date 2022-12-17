import datetime
import sqlite3
import timeit
import traceback
import uuid
from collections import namedtuple
from typing import get_args
from unittest import mock

import torch
from torch.multiprocessing import Process

from data import ActFn, Task, TrainConfig, sparsities
from toy_model import train_model
from training_db import insert_train_result

act_fns = get_args(ActFn)  #  ["ReLU", "GeLU", "SoLU"]


def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


SizeConfig = namedtuple("SizeConfig", ["neurons", "features"])

size_configs = [
    SizeConfig(3, 6),
    SizeConfig(5, 20),
    # SizeConfig(40, 100),
]

tasks = get_args(Task)


@mock.patch("tqdm.auto.tqdm", notqdm)
def train_one(act_fn, variation, s, size_config, task):
    try:
        start = timeit.default_timer()
        model_name = "HiddenLayerModelVariation" if variation else "HiddenLayerModel"
        config = TrainConfig(
            model_name,
            s=s,
            i=0.8,
            task=task,
            steps=100_000,
            act_fn=act_fn,
            args=dict(
                neurons=size_config.neurons,
                features=size_config.features,
            ),
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
        training_id = uuid.uuid4().hex
        print(
            f"{timestamp}: training {training_id}: {model_name}, {act_fn}, {s}, {size_config}, {task}"
        )
        train_result = train_model(config, device, checkpoint_every=100)
        stop = timeit.default_timer()
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")
        print(
            f"{timestamp}: inserting {training_id}: {model_name}, {act_fn}, {s}, {size_config}, {task} (after {stop - start}s)"
        )
        con = sqlite3.connect("hidden_layer.db")
        cur = con.cursor()
        insert_train_result(train_result, None, cur, con)
        train_result.save(f"train_results/{training_id}")
        with open("training_ids", "a") as myfile:
            myfile.write(training_id + "\n")
    except Exception:
        print(traceback.format_exc())
        raise


def train_some(config_batch):
    for config in config_batch:
        train_one(*config)


if __name__ == "__main__":
    configurations = [
        ("SoLU", False, s, SizeConfig(3, 6), "ABS")
        # for act_fn in ["ReLU", "GeLU", "SoLU"]  # act_fns
        # for variation in (False, True)
        for s in sparsities
        # for size_config in size_configs
        # for task in ["ID", "SQUARE", "MAX", "MIN"]
    ] * 4

    num_processes = 16
    work = []
    for i in range(num_processes):
        work.append([])
    for i in range(len(configurations)):
        work[i % num_processes].append(configurations[i])

    processes = []
    for config_batch in work:
        p = Process(target=train_some, args=(config_batch,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
