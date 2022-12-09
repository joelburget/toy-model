from dash import Dash, dcc, html, Input, Output
from toy_model import *
import plotly.graph_objects as go

app = Dash(__name__)

app.layout = html.Div(
    [
        html.H1("ReLU Variation"),
        html.Div(
            [
                f"Loss achieved: ",
                loss_achieved := html.Span(""),
            ]
        ),
        html.Div(
            [
                "Training run",
                run_selector := dcc.RadioItems([str(i) for i in range(20)], "0"),
            ]
        ),
        html.Div(
            [
                html.H2("Weights"),
                fig_sankey := dcc.Graph(
                    figure=go.Figure(),
                ),
            ]
        ),
        html.Div(
            [
                html.H2("Activations"),
                acts_sankey := dcc.Graph(
                    figure=go.Figure(),
                ),
                html.Div(
                    [
                        input_0 := dcc.Input(type="number", value=0),
                        input_1 := dcc.Input(type="number", value=1),
                        input_2 := dcc.Input(type="number", value=2),
                    ]
                ),
            ]
        ),
        html.H2("Weights"),
        html.Div(
            [
                fig_w1 := dcc.Graph(figure=go.Figure()),
                fig_w1_stack := dcc.Graph(figure=go.Figure()),
            ],
            style={"display": "flex", "flex-direction": "row"},
        ),
        html.Div(
            [
                fig_w2 := dcc.Graph(figure=go.Figure()),
                fig_w2_stack := dcc.Graph(figure=go.Figure()),
            ],
            style={"display": "flex", "flex-direction": "row"},
        ),
        html.Div(
            [
                fig_b := dcc.Graph(figure=go.Figure()),
            ],
        ),
    ],
    style={"display": "flex", "flex-direction": "column"},
)


def activations_sankey(layers: List[NDArray], inputs: NDArray):
    print(inputs)
    for x in inputs:
        if x is None:
            return go.Figure(layout_title_text="(invalid input)")

    nodes_seen = 0
    sources, targets, values, colors, labels, node_names = [], [], [], [], [], []
    xs, ys = [], []

    node_names += [str(x) for x in inputs]

    xs_space = np.linspace(0.01, 0.99, len(layers) + 1)
    xs += [xs_space[0]] * len(inputs)
    ys += list(np.linspace(0.01, 0.99, len(inputs)))
    prev_layer_acts = inputs

    for layer_num, layer in enumerate(layers):
        print(prev_layer_acts)
        n_rows, n_cols = layer.shape
        activations = layer @ prev_layer_acts
        node_names += [str(x) for x in activations]
        xs += [xs_space[layer_num + 1]] * n_rows
        ys += list(np.linspace(0.01, 0.99, n_rows))
        for i in range(n_cols):
            for j in range(n_rows):
                sources.append(nodes_seen + i)
                targets.append(nodes_seen + len(prev_layer_acts) + j)
                values.append(abs(layer[j, i] * prev_layer_acts[i]))
                labels.append(f"{layer[j, i]:.2f}")
                colors.append(
                    "rgba(255,0,0, 0.3)" if layer[j, i] > 0 else "rgba(0,0,255, 0.3)"
                )
        nodes_seen += len(prev_layer_acts)
        prev_layer_acts = activations

    return go.Figure(
        layout_title_text="Activations for Given Inputs",
        data=[
            go.Sankey(
                node=dict(
                    label=node_names,
                    x=xs,
                    y=ys,
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    label=labels,
                    color=colors,
                ),
            )
        ],
    )


@app.callback(
    Output(fig_sankey, "figure"),
    Output(acts_sankey, "figure"),
    Output(fig_w1, "figure"),
    Output(fig_w1_stack, "figure"),
    Output(fig_w2, "figure"),
    Output(fig_w2_stack, "figure"),
    Output(fig_b, "figure"),
    Output(loss_achieved, "children"),
    Input(run_selector, "value"),
    Input(input_0, "value"),
    Input(input_1, "value"),
    Input(input_2, "value"),
)
def fill_values(run_num, input_0, input_1, input_2):
    train_result = TrainResult.load(f"relu-variation/relu-variation-{run_num}")
    model = train_result.config.model
    figs = ReluHiddenLayerModelVariation.plots(train_result)
    return (
        weights_sankey([model.W1.detach().numpy(), model.W2.detach().numpy()]),
        activations_sankey(
            [model.W1.detach().numpy(), model.W2.detach().numpy()],
            np.array([input_0, input_1, input_2]),
        ),
        *figs,
        f"{train_result.losses[-1]:.8f}",
    )


if __name__ == "__main__":
    app.run_server(debug=True)
