import json
import sqlite3

import dash_daq as daq
import plotly.express as px
import plotly.graph_objects as go
import torch
from dash import Dash, Input, Output, dcc, html

import training
from data import sparsities
from training_db import get_result

app = Dash(__name__)


def rows(children):
    return html.Div(
        children,
        style={"display": "flex", "flex-direction": "column"},
    )


def cols(children):
    return html.Div(
        children,
        style={"display": "flex", "flex-direction": "row"},
    )


def foldable_section(title, *children):
    return html.Details(
        [
            html.Summary(title),
            *children,
        ],
    )


def slider():
    return daq.Slider(min=-10, max=10, value=0, marks={"0": "0"})


app.layout = rows(
    [
        html.H1("Toy Model Selector"),
        rows(
            [
                cols(
                    [
                        "Activation Function",
                        act_fn_selector := dcc.RadioItems(training.act_fns, "ReLU"),
                    ]
                ),
                cols(
                    [
                        "Sparsity",
                        sparsity_selector := dcc.RadioItems(
                            [str(s) for s in sparsities], "0"
                        ),
                    ]
                ),
                cols(
                    [
                        "Task",
                        task_selector := dcc.RadioItems(
                            ["ID", "SQUARE", "MAX", "MIN"], "ID"
                        ),
                    ]
                ),
                cols(
                    [
                        "Run number",
                        rows(
                            [
                                run_no_selector := dcc.RadioItems(
                                    [],
                                    "0",
                                    style={
                                        "display": "flex",
                                        "flex-direction": "column",
                                    },
                                ),
                                avg_loss := html.Span(""),
                            ]
                        ),
                    ]
                ),
            ]
        ),
        foldable_section(
            "Loss",
            fig_loss := dcc.Graph(figure=go.Figure()),
        ),
        foldable_section(
            "Activations Graph",
            rows(
                [
                    input0 := slider(),
                    input1 := slider(),
                    input2 := slider(),
                    input3 := slider(),
                    input4 := slider(),
                    input5 := slider(),
                    input6 := slider(),
                    input7 := slider(),
                    input8 := slider(),
                    input9 := slider(),
                    input10 := slider(),
                    input11 := slider(),
                    input12 := slider(),
                    input13 := slider(),
                    input14 := slider(),
                    input15 := slider(),
                    input16 := slider(),
                    input17 := slider(),
                    input18 := slider(),
                    input19 := slider(),
                ],
            ),
            fig_activations_graph := dcc.Graph(figure=go.Figure()),
        ),
        foldable_section(
            "Weights",
            cols(
                [
                    fig_w_square := dcc.Graph(figure=go.Figure()),
                    fig_w := dcc.Graph(figure=go.Figure()),
                ]
            ),
        ),
        ln_plots := html.Div(),
        foldable_section(
            "Sankey Diagram",
            fig_weights_sankey := dcc.Graph(figure=go.Figure()),
        ),
        foldable_section(
            "Bias",
            fig_b := dcc.Graph(figure=go.Figure()),
        ),
        train_result_nums := dcc.Store(id="train-result-nums"),
        train_result_no := dcc.Store(id="train-result-no"),
    ]
)


@app.callback(
    Output(train_result_nums, "data"),
    Input(act_fn_selector, "value"),
    Input(sparsity_selector, "value"),
    Input(task_selector, "value"),
)
def update_run_nums(act_fn, sparsity, task_name):
    con = sqlite3.connect("new.db")
    cur = con.cursor()
    cur.execute(
        """SELECT run_no FROM training_run
           WHERE name = 'ToyModel'
           AND act_fn = :act_fn
           AND sparsity = :sparsity
           AND task = :task_name
        """,
        dict(act_fn=act_fn, sparsity=sparsity, task_name=task_name),
    )

    run_nos = cur.fetchall()
    return json.dumps([i for (i,) in run_nos])


@app.callback(
    Output(run_no_selector, "options"),
    Output(run_no_selector, "value"),
    Output(avg_loss, "children"),
    Input(train_result_nums, "data"),
)
def update_run_selector(train_result_nums):
    run_nums = json.loads(train_result_nums)

    if len(run_nums) == 0:
        return [], 0, ""

    all_losses = dict()
    for i in run_nums:
        train_result = get_result(i)
        all_losses[i] = train_result.losses[-1]
    avg_loss = sum(all_losses) / len(all_losses)

    # print(f"run_nums: {run_nums}")
    options = [dict(label=f"{i}: {all_losses[i]:.3f}", value=i) for i in run_nums]

    return (
        options,
        run_nums[0],
        f"average loss {avg_loss:.3f}",
    )


@app.callback(
    Output(train_result_no, "data"),
    Input(train_result_nums, "data"),
    Input(run_no_selector, "value"),
)
def get_model(train_result_nums, run_num):
    run_nums = json.loads(train_result_nums)
    # print(f"train_result_nums: {train_result_nums}")
    # print(f"run_nums: {run_nums}")

    if len(run_nums) == 0:
        return 1

    # print(f"train_result_no: {run_num}")
    return int(run_num)


@app.callback(
    Output(fig_w_square, "figure"),
    Output(fig_w, "figure"),
    Output(fig_b, "figure"),
    Output(fig_weights_sankey, "figure"),
    Output(ln_plots, "children"),
    Output(fig_loss, "figure"),
    Input(train_result_no, "data"),
)
def update_plots(train_result_no):
    train_result = get_result(int(train_result_no))

    with torch.no_grad():
        model = train_result.model.cpu()
    plots = model.plots()

    ln_plots = None
    if "ln_w" in plots:
        ln_plots = foldable_section(
            "LayerNorm",
            cols(
                [
                    dcc.Graph(figure=plots["ln_w"]),
                    dcc.Graph(figure=plots["ln_b"]),
                ]
            ),
        )

    return (
        plots["w_square"],
        plots["w"],
        plots["b"],
        plots["weights_sankey"],
        ln_plots,
        px.scatter(train_result.losses, log_y=True),
    )


@app.callback(
    Output(fig_activations_graph, "figure"),
    Input(input0, "value"),
    Input(input1, "value"),
    Input(input2, "value"),
    Input(input3, "value"),
    Input(input4, "value"),
    Input(input5, "value"),
    Input(input6, "value"),
    Input(input7, "value"),
    Input(input8, "value"),
    Input(input9, "value"),
    Input(input10, "value"),
    Input(input11, "value"),
    Input(input12, "value"),
    Input(input13, "value"),
    Input(input14, "value"),
    Input(input15, "value"),
    Input(input16, "value"),
    Input(input17, "value"),
    Input(input18, "value"),
    Input(input19, "value"),
    Input(train_result_no, "data"),
)
def update_activations_graph(
    input0,
    input1,
    input2,
    input3,
    input4,
    input5,
    input6,
    input7,
    input8,
    input9,
    input10,
    input11,
    input12,
    input13,
    input14,
    input15,
    input16,
    input17,
    input18,
    input19,
    train_result_no,
):
    train_result = get_result(int(train_result_no))
    input = torch.zeros((1, 20))
    input[0] = torch.tensor(
        [
            input0,
            input1,
            input2,
            input3,
            input4,
            input5,
            input6,
            input7,
            input8,
            input9,
            input10,
            input11,
            input12,
            input13,
            input14,
            input15,
            input16,
            input17,
            input18,
            input19,
        ]
    )
    with torch.no_grad():
        model = train_result.model.cpu()
    return model.activations_graph(input)


if __name__ == "__main__":
    app.run_server(debug=True, host="192.168.86.201")
