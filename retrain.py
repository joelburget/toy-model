import datetime
import sqlite3
import timeit
from multiprocessing import Pool
from unittest import mock

import torch

from toy_model import retrain_model
from training_db import get_result


def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


@mock.patch("tqdm.auto.tqdm", notqdm)
def train_one(n):
    start = timeit.default_timer()
    print(f"starting {n}")

    previous_result = get_result(n, load_checkpoints=True)
    config = previous_result.config
    model = previous_result.model
    checkpoints = previous_result.checkpoints

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_result = retrain_model(model, config, device, checkpoints=checkpoints)

    stop = timeit.default_timer()
    print(f"{n} complete (after {stop - start}s), updating")

    con = sqlite3.connect("training.db")
    cur = con.cursor()
    cur.execute(
        """UPDATE training_run
           SET steps = :steps, created = :created
           WHERE run_no = :run_no
        """,
        # previously trained for 50_000, config also set to 50_000
        dict(steps=100_000, created=datetime.datetime.now(), run_no=n),
    )
    train_result.save(f"train_results/{n}")


def get_run_nos():
    con = sqlite3.connect("training.db")
    cur = con.cursor()
    cur.execute('SELECT run_no FROM training_run WHERE name = "LayerNormToyModel"')
    return [num for (num,) in cur.fetchall()]


if __name__ == "__main__":
    run_nos = get_run_nos()

    with Pool(processes=16) as pool:
        pool.map(train_one, run_nos)
        pool.close()
        pool.join()
