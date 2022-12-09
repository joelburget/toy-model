import sqlite3
from data import TrainConfig, TrainResult

con = sqlite3.connect("training.db")
cur = con.cursor()


def initialize_db():
    cur.execute(
        """CREATE TABLE training_run (
               run_no,
               name,
               sparsity,
               importance,
               points,
               steps,
               task,
               regularization_coeff
           )
        """
    )


def get_next_num() -> int:
    (count,) = cur.execute("SELECT COUNT(*) FROM training_run").fetchone()
    return count + 1


def insert_conf(conf: TrainConfig):
    cur.execute(
        "INSERT INTO training_run VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
        (
            get_next_num(),
            conf.model_name,
            conf.s,
            conf.i,
            conf.points,
            conf.steps,
            conf.task,
            conf.regularization_coeff,
        ),
    )
    con.commit()


def insert_train_result(train_result):
    n = get_next_num()
    train_result.save(f"train_results/{n}")
    insert_conf(train_result.config)


def get_config(n: int) -> TrainConfig:
    res = cur.execute("SELECT * FROM training_run WHERE run_no = ?", (n,))
    _, name, s, i, points, steps, task, regularization_coeff = res.fetchone()
    return TrainConfig(name, s, i, points, steps, task, regularization_coeff)


def get_result(n: int) -> TrainResult:
    return TrainResult.load(f"train_results/{n}")
