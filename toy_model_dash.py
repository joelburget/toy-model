from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px
import training
from data import sparsities
import sqlite3
from training_db import get_result
import torch

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("Toy Model Selector"),
        html.Ul(
            [
                html.Li(
                    [
                        "Activation Function",
                        act_fn_selector := dcc.RadioItems(training.act_fns, "ReLU"),
                    ]
                ),
                html.Li(
                    [
                        "Run number",
                        run_no_selector := dcc.RadioItems(
                            [str(i) for i in range(5)], "0"
                        ),
                        run_losses := html.Span(""),
                    ]
                ),
                html.Li(
                    [
                        "Sparsity",
                        sparsity_selector := dcc.RadioItems(
                            [str(s) for s in sparsities], "0"
                        ),
                    ]
                ),
                html.Li(
                    [
                        "Model",
                        model_selector := dcc.RadioItems(
                            ["ToyModel", "LayerNormToyModel"], "ToyModel"
                        ),
                    ]
                ),
            ]
        ),
        html.H2("Loss"),
        html.Div(
            [
                fig_loss := dcc.Graph(figure=go.Figure()),
            ],
            style={"display": "flex", "flex-direction": "row"},
        ),
        html.H2("Weights"),
        html.Div(
            [
                fig_w_square := dcc.Graph(figure=go.Figure()),
                fig_w := dcc.Graph(figure=go.Figure()),
            ],
            style={"display": "flex", "flex-direction": "row"},
        ),
        ln_plots := html.Div(),
        html.Div(
            [
                fig_b := dcc.Graph(figure=go.Figure()),
            ],
        ),
    ],
    style={"display": "flex", "flex-direction": "column"},
)


# def activations_sankey(layers: List[NDArray], inputs: NDArray):
#     for x in inputs:
#         if x is None:
#             return go.Figure(layout_title_text="(invalid input)")

#     nodes_seen = 0
#     sources, targets, values, node_names = [], [], [], []

#     node_names += [f"I{i}={x:.3f}" for i, x in enumerate(inputs)]
#     prev_layer_acts = inputs

#     for layer_num, layer in enumerate(layers):
#         n_rows, n_cols = layer.shape
#         activations = layer @ prev_layer_acts
#         layer_name = chr(layer_num + ord("A"))
#         node_names += [f"{layer_name}{i}={x:.3f}" for i, x in enumerate(activations)]
#         for i in range(n_cols):
#             for j in range(n_rows):
#                 sources.append(nodes_seen + i)
#                 targets.append(nodes_seen + len(prev_layer_acts) + j)
#                 values.append(layer[j, i] * prev_layer_acts[i])
#         nodes_seen += len(prev_layer_acts)
#         prev_layer_acts = activations

#     # XXX transpose?
#     layer_shapes = [layers[0].shape[1], layers[0].shape[0]]
#     layer_shapes += [layer.shape[0] for layer in layers[1:]]

#     return sankey(
#         "Activations for Given Inputs",
#         sources,
#         targets,
#         values,
#         node_names,
#         layer_shapes,
#     )


@app.callback(
    # Output(fig_sankey, "figure"),
    Output(fig_w_square, "figure"),
    Output(fig_w, "figure"),
    Output(fig_b, "figure"),
    Output(ln_plots, "children"),
    Output(fig_loss, "figure"),
    Output(run_losses, "children"),
    Input(act_fn_selector, "value"),
    Input(run_no_selector, "value"),
    Input(sparsity_selector, "value"),
    Input(model_selector, "value"),
)
def update_values(act_fn, run_num, sparsity, model_name):
    con = sqlite3.connect("training.db")
    cur = con.cursor()

    cur.execute(
        """SELECT run_no FROM training_run
           WHERE act_fn = :act_fn
           AND sparsity = :sparsity
           AND name = :model_name
        """,
        dict(act_fn=act_fn, sparsity=sparsity, model_name=model_name),
    )
    run_nos = cur.fetchall()
    run_nos = [i for (i,) in run_nos]

    all_losses = []
    for i in run_nos:
        train_result = get_result(i)
        all_losses.append(train_result.losses[-1])
    avg_loss = sum(all_losses) / len(all_losses)
    formatted_losses = ", ".join([f"{x:.3f}" for x in all_losses])

    train_result_no = run_nos[int(run_num)]
    train_result = get_result(int(train_result_no))
    with torch.no_grad():
        train_result.model.cpu()
        plots = train_result.model.plots()

    ln_plots = None
    if "ln_w" in plots:
        ln_plots = html.Div(
            [
                html.H2("LayerNorm"),
                html.Div(
                    [
                        dcc.Graph(figure=plots["ln_w"]),
                        dcc.Graph(figure=plots["ln_b"]),
                    ],
                    style={
                        "display": "flex",
                        "flex-direction": "row",
                        "height": "400px",
                    },
                ),
            ],
            style={"display": "flex", "flex-direction": "column"},
        )

    return (
        # weights_sankey([model.W1.detach().numpy(), model.W2.detach().numpy()]),
        plots["w_square"],
        plots["w"],
        plots["b"],
        ln_plots,
        px.scatter(train_result.losses, log_y=True),
        f"losses: {formatted_losses} (average {avg_loss:.3f})",
    )


if __name__ == "__main__":
    app.run_server(debug=True, host="192.168.86.201")
