import json
import os
import sqlite3

import dash_daq as daq
import numpy as np
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
                        "Model Name",
                        model_name_selector := dcc.RadioItems(
                            ["HiddenLayerModel", "HiddenLayerModelVariation"],
                            "HiddenLayerModel",
                        ),
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
                            ["ID", "SQUARE", "MAX", "MIN", "ABS"], "ABS"
                        ),
                    ]
                ),
                cols(
                    [
                        "Run number",
                        rows(
                            [
                                run_id_selector := dcc.RadioItems(
                                    [],
                                    "",
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
                    fig_w1 := dcc.Graph(figure=go.Figure()),
                    fig_w2 := dcc.Graph(figure=go.Figure()),
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
        foldable_section(
            "Checkpoints",
            checkpoint_view := dcc.Graph(figure=go.Figure()),
        ),
        train_result_ids := dcc.Store(id="train-result-ids"),
        train_result_id := dcc.Store(id="train-result-no"),
    ]
)


@app.callback(
    Output(train_result_ids, "data"),
    Input(model_name_selector, "value"),
    Input(act_fn_selector, "value"),
    Input(sparsity_selector, "value"),
    Input(task_selector, "value"),
)
def update_run_ids(model_name, act_fn, sparsity, task_name):
    con = sqlite3.connect("hidden_layer.db")
    cur = con.cursor()
    cur.execute(
        """SELECT run_no FROM training_run
           WHERE name = :model_name
           AND act_fn = :act_fn
           AND sparsity = :sparsity
           AND task = :task_name
        """,
        dict(
            model_name=model_name, act_fn=act_fn, sparsity=sparsity, task_name=task_name
        ),
    )

    run_ids = cur.fetchall()
    run_ids = [s for (s,) in run_ids]
    print(run_ids)
    # run_nos = set(run_nos)

    # deduplicate based on final loss
    # all_losses = dict()
    # for i in run_nos:
    #     train_result = get_result(i)
    #     all_losses[train_result.losses[-1]] = i
    # run_nos = all_losses.values()

    return json.dumps(run_ids)


@app.callback(
    Output(run_id_selector, "options"),
    Output(run_id_selector, "value"),
    Output(avg_loss, "children"),
    Input(train_result_ids, "data"),
)
def update_run_selector(train_result_ids):
    run_ids = json.loads(train_result_ids)

    if len(run_ids) == 0:
        return [], "", ""

    all_losses = dict()
    params = dict()
    has_checkpoint = dict()
    for run_id in run_ids:
        train_result = get_result(run_id, load_checkpoints=False)
        args = train_result.config.args
        all_losses[run_id] = train_result.losses[-1]
        params[run_id] = args["features"], args["neurons"]
        has_checkpoint[run_id] = os.path.exists(
            f"train_results/{run_id}/checkpoints.pkl"
        )
    avg_loss = sum(all_losses.values()) / len(all_losses)

    options = [
        (
            all_losses[run_id],
            dict(
                label=f"{run_id[:4]}{'*' if has_checkpoint[run_id] else ''} {params[run_id]}: {all_losses[run_id]:.3f}",
                value=run_id,
            ),
        )
        for run_id in run_ids
    ]
    options.sort(key=lambda x: x[0])
    options = [x[1] for x in options]

    return (
        options,
        options[0]["value"],
        f"average loss {avg_loss:.3f}",
    )


@app.callback(
    Output(train_result_id, "data"),
    Input(train_result_ids, "data"),
    Input(run_id_selector, "value"),
)
def get_run_id(train_result_ids, run_id):
    run_ids = json.loads(train_result_ids)

    if len(run_ids) == 0:
        return ""

    return run_id


@app.callback(
    Output(checkpoint_view, "figure"),
    Input(train_result_id, "data"),
)
def update_checkpoint_view(train_result_id):
    if not train_result_id:
        return go.Figure()

    train_result = get_result(train_result_id, load_checkpoints=True)
    model = train_result.model
    model.cpu()
    checkpoints = train_result.checkpoints

    if not checkpoints:
        return go.Figure()

    ims = np.empty(
        (len(checkpoints) // 10, model.features, model.features), dtype=np.float32
    )

    with torch.no_grad():
        for i, cpkt in enumerate(checkpoints):
            if i % 10 == 0:
                model.load_state_dict(cpkt[1])
                w = model.W.numpy()
                ims[i // 10] = w.T @ w

    return px.imshow(ims, animation_frame=0, title="w.T @ w")


@app.callback(
    Output(fig_w1, "figure"),
    Output(fig_w2, "figure"),
    Output(fig_b, "figure"),
    Output(fig_weights_sankey, "figure"),
    Output(ln_plots, "children"),
    Output(fig_loss, "figure"),
    Input(train_result_id, "data"),
)
def update_plots(train_result_id):
    if not train_result_id:
        return (
            go.Figure(),
            go.Figure(),
            go.Figure(),
            go.Figure(),
            [],
            go.Figure(),
        )

    train_result = get_result(train_result_id, load_checkpoints=False)

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
        plots.get("w", plots.get("w1", None)),
        plots.get("w2", go.Figure()),
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
    Input(train_result_id, "data"),
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
    train_result_id,
):
    if not train_result_id:
        return go.Figure()
    train_result = get_result(train_result_id, load_checkpoints=False)
    features = train_result.model.features
    input = torch.zeros((1, features))
    inputs = [
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
    inputs = inputs[:features]
    input[0][0 : len(inputs)] = torch.tensor(inputs)
    with torch.no_grad():
        model = train_result.model.cpu()
    model.eval()
    return model.activations_graph(input)


if __name__ == "__main__":
    app.run_server(debug=True, host="192.168.86.201")
