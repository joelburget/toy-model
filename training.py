from toy_model import train_model
from data import TrainConfig, sparsities
from training_db import insert_train_result
from multiprocessing import Pool, Lock
import timeit
from unittest import mock
import torch
from collections import namedtuple

act_fns = ["ReLU", "GeLU", "SoLU"]
DUPS = 5


def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


SizeConfig = namedtuple("SizeConfig", ["neurons", "features"])

size_configs = [
    # (5, 20),  # 5 neurons, 20 features, already trained
    SizeConfig(40, 100),
]


@mock.patch("tqdm.auto.tqdm", notqdm)
def train_one(act_fn, use_ln, s, size_config):
    start = timeit.default_timer()
    print(f"starting {act_fn}, {use_ln}, {s}")
    config = TrainConfig(
        "LayerNormToyModel" if use_ln else "ToyModel",
        s=s,
        i=0.8,
        task="ID",
        steps=(100_000 if use_ln else 50_000),
        act_fn=act_fn,
        args=dict(
            neurons=size_config.neurons,
            features=size_config.features,
        ),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_result = train_model(config, device)
    stop = timeit.default_timer()
    print(f"inserting {act_fn}, {use_ln}, {s} (after {stop - start}s)")
    insert_train_result(train_result, lock)


def init_pool_processes(the_lock):
    """Initialize each process with a global variable lock."""
    global lock
    lock = the_lock


if __name__ == "__main__":
    lock = Lock()
    configurations = [
        (act_fn, use_ln, s, size_config)
        for act_fn in act_fns
        for use_ln in (False, True)
        for s in sparsities
        for size_config in size_configs
    ] * DUPS

    with Pool(initializer=init_pool_processes, initargs=(lock,)) as pool:
        pool.starmap(train_one, configurations)
        pool.close()
        pool.join()
