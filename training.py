from toy_model import train_model
from data import TrainConfig, sparsities
from training_db import insert_train_result
from multiprocessing import Pool, Lock
import timeit
from unittest import mock
import torch

act_fns = ["ReLU", "GeLU", "SoLU"]
DUPS = 5


def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


@mock.patch("tqdm.auto.tqdm", notqdm)
def go(act_fn, use_ln, s):
    start = timeit.default_timer()
    print(f"starting {act_fn}, {use_ln}, {s}")
    config = TrainConfig(
        "ToyModel" if not use_ln else "LayerNormToyModel",
        s=s,
        i=0.8,
        task="ID",
        steps=50_000,
        act_fn=act_fn,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_result = train_model(config, device)
    stop = timeit.default_timer()
    print(f"{act_fn}, {use_ln}, {s} complete (after {stop - start}s), inserting")
    insert_train_result(train_result, lock)


def init_pool_processes(the_lock):
    """Initialize each process with a global variable lock."""
    global lock
    lock = the_lock


if __name__ == "__main__":
    lock = Lock()
    configurations = [
        (act_fn, use_ln, s)
        for act_fn in act_fns
        for use_ln in (False, True)
        for s in sparsities
    ] * DUPS

    with Pool(processes=16, initializer=init_pool_processes, initargs=(lock,)) as pool:
        pool.starmap(go, configurations)
        pool.close()
        pool.join()
