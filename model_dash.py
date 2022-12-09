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
    for x in inputs:
        if x is None:
            return go.Figure(layout_title_text="(invalid input)")

    nodes_seen = 0
    sources, targets, values, node_names = [], [], [], []

    node_names += [f"I{i}={x:.3f}" for i, x in enumerate(inputs)]
    prev_layer_acts = inputs

    for layer_num, layer in enumerate(layers):
        n_rows, n_cols = layer.shape
        activations = layer @ prev_layer_acts
        layer_name = chr(layer_num + ord("A"))
        node_names += [f"{layer_name}{i}={x:.3f}" for i, x in enumerate(activations)]
        for i in range(n_cols):
            for j in range(n_rows):
                sources.append(nodes_seen + i)
                targets.append(nodes_seen + len(prev_layer_acts) + j)
                values.append(layer[j, i] * prev_layer_acts[i])
        nodes_seen += len(prev_layer_acts)
        prev_layer_acts = activations

    # XXX transpose?
    layer_shapes = [layers[0].shape[1], layers[0].shape[0]]
    layer_shapes += [layer.shape[0] for layer in layers[1:]]

    return sankey(
        "Activations for Given Inputs",
        sources,
        targets,
        values,
        node_names,
        layer_shapes,
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
