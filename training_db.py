import datetime
import json
import sqlite3

from data import TrainConfig, TrainResult

global_con = sqlite3.connect("training.db")
global_cur = global_con.cursor()


def initialize_db(cur=global_cur):
    cur.execute(
        """CREATE TABLE training_run (
               run_no               INTEGER,
               name                 TEXT,
               sparsity             REAL,
               importance           REAL,
               points               INTEGER,
               steps                INTEGER,
               task                 STRING,
               regularization_coeff REAL,
               act_fn               TEXT,
               args                 TEXT,
               created              TIMESTAMP
           )
        """
    )


def get_next_num(cur=global_cur) -> int:
    (count,) = cur.execute("SELECT COUNT(*) FROM training_run").fetchone()
    return count + 1


def insert_conf(conf: TrainConfig, cur=global_cur, con=global_con):
    cur.execute(
        """INSERT INTO training_run VALUES (
               :run_no,
               :name,
               :sparsity,
               :importance,
               :points,
               :steps,
               :task,
               :regularization_coeff,
               :act_fn,
               :args,
               :created
            )
        """,
        dict(
            run_no=get_next_num(cur),
            name=conf.model_name,
            sparsity=conf.s,
            importance=conf.i,
            points=conf.points,
            steps=conf.steps,
            task=conf.task,
            regularization_coeff=conf.regularization_coeff,
            act_fn=conf.act_fn,
            args=json.dumps(conf.args),
            created=datetime.datetime.now(),
        ),
    )
    con.commit()


def insert_train_result(train_result, lock=None, cur=global_cur, con=global_con):
    try:
        if lock:
            lock.acquire()
        n = get_next_num()
        train_result.save(f"train_results/{n}")
        insert_conf(train_result.config, cur=cur, con=con)
    finally:
        if lock:
            lock.release()


def populate_train_config(row):
    (
        _,
        name,
        s,
        i,
        points,
        steps,
        task,
        regularization_coeff,
        act_fn,
        args,
        _,
    ) = row
    return TrainConfig(
        name, s, i, points, steps, task, regularization_coeff, act_fn, json.loads(args)
    )


def get_config(n: int, cur=global_cur) -> TrainConfig:
    res = cur.execute("SELECT * FROM training_run WHERE run_no = ?", (n,))
    populate_train_config(res.fetchone())


def get_result(n: int) -> TrainResult:
    return TrainResult.load(f"train_results/{n}")


if __name__ == "__main__":
    initialize_db()
