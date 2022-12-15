import math
from functools import partial
from typing import Callable, List

import einops
import numpy as np
import pandas
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm.auto as tqdm
from colour import Color
from numpy.typing import NDArray

from data import ActFn, TrainConfig, TrainResult


def solu(x: torch.Tensor) -> torch.Tensor:
    return x * F.softmax(x, dim=1)


def act_fn(name: ActFn) -> Callable[[torch.Tensor], torch.Tensor]:
    if name == "ReLU":
        return F.relu
    elif name == "GeLU":
        return F.gelu
    else:
        return solu


def create_model(config: TrainConfig) -> nn.Module:
    if config.model_name == "ToyModel":
        model_class = ToyModel
    elif config.model_name == "LayerNormToyModel":
        model_class = LayerNormToyModel
    elif config.model_name in ("ReluHiddenLayerModel", "HiddenLayerModel"):
        model_class = HiddenLayerModel
    elif config.model_name in (
        "ReluHiddenLayerModelVariation",
        "HiddenLayerModelVariation",
    ):
        model_class = HiddenLayerModelVariation
    # if config.model_name == "MultipleHiddenLayerModel":
    #     model_class = MultipleHiddenLayerModel
    elif config.model_name == "MlpModel":
        model_class = MlpModel
    else:
        model_class = ResidualModel

    return model_class(act_fn=config.act_fn, **config.args)


# mean squared error weighted by feature importance
def loss_fn(importance, y_pred, y_true, device="cpu"):
    _, features = y_pred.shape
    error = y_true - y_pred
    # importance falls off geometrically
    importance = torch.tensor([importance**i for i in range(features)]).to(device)
    return torch.mean(error**2 * importance)


def pairwise_op(op, t):
    t = einops.reduce(t, "batch (d 2) -> batch d", "max")
    return einops.repeat(t, "batch d -> batch (d 2)")


def train_model(config: TrainConfig, device="cpu"):
    return retrain_model(create_model(config), config, device)


def retrain_model(model: nn.Module, config: TrainConfig, device="cpu"):
    features: int = model.features
    lower_bound, upper_bound = -10, 10

    model.to(device)

    # Start with random data
    x_train = torch.FloatTensor(config.points, features).uniform_(
        lower_bound, upper_bound
    )
    x_test = torch.FloatTensor(config.points * 2, features).uniform_(
        lower_bound, upper_bound
    )

    # Then apply sparsity
    for x in [x_train, x_test]:
        t = config.s * torch.ones(len(x), features)
        mask = torch.bernoulli(t) > 0
        x[mask] = 0

    # Remove points that are completely zero
    x_train = x_train[(x_train == 0).sum(axis=1) != features].to(device)
    x_test = x_test[(x_test == 0).sum(axis=1) != features].to(device)

    # Train model
    optimizer = torch.optim.AdamW(model.parameters())

    task = lambda x: x
    if config.task == "SQUARE":
        task = lambda x: x**2
    elif config.task == "ABS":
        task = lambda x: abs(x)
    elif config.task == "MAX":
        task = partial(pairwise_op, "max")
    elif config.task == "MIN":
        task = partial(pairwise_op, "min")

    losses = []
    for t in tqdm.tqdm(range(config.steps)):
        prediction, _ = model(x_train)
        actual = task(x_train)
        loss = (
            loss_fn(config.i, prediction, actual, device=device)
            # + config.regularization_coeff * l1_terms.abs().sum()
        )
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return TrainResult(model, config, losses, x_train, x_test)


def stack_plot_df(w):
    _, cols = w.shape
    fig = px.bar(
        w, x="neuron", y=list(str(i) for i in range(cols - 1)), title="Stack Plot"
    )
    # fig.update(data=[{'hovertemplate':"neuron: %{x}<br />feature: %{variable}<br />value: %{value:.4f}"}])
    return fig


def mk_dataframe(w):
    n_neurons, n_features = w.shape
    w = np.concatenate([np.array(range(n_neurons))[:, None], w], axis=1)
    return pandas.DataFrame(
        w, columns=["neuron"] + list(str(i) for i in range(n_features))
    )


def stack_plot(w):
    """w: neurons x features"""
    return stack_plot_df(mk_dataframe(w))


def generic_sankey(title: str, sources, targets, values, node_names, layer_shapes):
    colors, labels, xs, ys = [], [], [], []

    most_neg, most_pos = math.inf, -math.inf
    for v in values:
        most_neg = min(v, most_neg)
        most_pos = max(v, most_pos)

    color_range = most_pos - most_neg
    color_options = list(Color("blue").range_to(Color("red"), 100))

    for i in range(len(values)):
        color = color_options[
            math.floor((values[i] - most_neg) * 99 / color_range)
        ].hex_l
        colors.append(color)
        labels.append(f"{values[i]:.3f}")
        values[i] = abs(values[i])

        # sankey won't show the node if all of its outputs are 0 :facepalm:
        if values[i] == 0.0:
            values[i] += 1e-4

    xs_space = np.linspace(0.01, 0.99, len(layer_shapes))
    for layer_num, n_rows in enumerate(layer_shapes):
        xs += [xs_space[layer_num]] * n_rows
        ys += list(np.linspace(0.01, 0.99, n_rows))

    return go.Figure(
        layout_title_text=title,
        data=[
            go.Sankey(
                arrangement="fixed",
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


def activations_sankey(layers: List[torch.tensor]):
    nodes_seen = 0
    sources, targets, values, node_names = [], [], [], []

    # node_names += [f"I{i}={x:.3f}" for i, x in enumerate(inputs)]
    # prev_layer_acts = inputs

    for layer_num, layer in enumerate(layers):
        n_rows, n_cols = layer.shape
        # activations = layer @ prev_layer_acts
        layer_name = chr(layer_num + ord("A"))
        node_names += [f"{layer_name}{i}={x:.3f}" for i, x in enumerate(activations)]
        for i in range(n_cols):
            for j in range(n_rows):
                sources.append(nodes_seen + i)
                # targets.append(nodes_seen + len(prev_layer_acts) + j)
                # values.append(layer[j, i] * prev_layer_acts[i])
        # nodes_seen += len(prev_layer_acts)
        # prev_layer_acts = activations

    # XXX transpose?
    layer_shapes = [layers[0].shape[1], layers[0].shape[0]]
    layer_shapes += [layer.shape[0] for layer in layers[1:]]

    return generic_sankey(
        "Activations for Given Inputs",
        sources,
        targets,
        values,
        node_names,
        layer_shapes,
    )


def weights_sankey(layers: List[NDArray]):
    # layers: e.g. [m, n], [n, o]
    # e.g. [neurons, features], [features, neurons]

    nodes_seen = 0
    sources, targets, values, node_names = [], [], [], []

    n_inputs = layers[0].shape[1]
    node_names += ["In" + str(n) for n in range(n_inputs)]
    prev_layer_nodes = n_inputs

    for layer_num, layer in enumerate(layers):
        layer_name = chr(layer_num + ord("A"))
        n_rows, n_cols = layer.shape
        node_names += [layer_name + str(j) for j in range(n_rows)]
        for i in range(n_cols):
            for j in range(n_rows):
                sources.append(nodes_seen + i)
                targets.append(nodes_seen + prev_layer_nodes + j)
                values.append(layer[j, i])
        nodes_seen += prev_layer_nodes
        prev_layer_nodes = n_rows

    # XXX transpose?
    layer_shapes = [layers[0].shape[1], layers[0].shape[0]]
    layer_shapes += [layer.shape[0] for layer in layers[1:]]

    return generic_sankey("Weights", sources, targets, values, node_names, layer_shapes)


def activations_graph(layer_cache):
    layer_node_count = [len(layer[0]) for layer in layer_cache]

    node_nums = []
    for layer in layer_node_count:
        node_count = sum(len(layer) for layer in node_nums)
        node_nums.append([i + node_count for i in range(layer)])

    def connected(l1, l2):
        return [(i, j) for i in node_nums[l1] for j in node_nums[l2]]

    def pointwise(l1, l2):
        return [(node_nums[l1][i], node_nums[l2][i]) for i in range(len(node_nums[l1]))]

    edges = connected(0, 1) + connected(1, 2) + pointwise(2, 3) + pointwise(3, 4)

    most_neg, most_pos = math.inf, -math.inf
    for layer in layer_cache:
        for v in layer[0]:
            most_neg = min(v.item(), most_neg)
            most_pos = max(v.item(), most_pos)

    color_range = most_pos - most_neg
    color_options = list(Color("blue").range_to(Color("red"), 100))

    activations = []
    colors = []
    for layer in layer_cache:
        activations += [v.item() for v in layer[0]]
        for v in layer[0]:
            colors.append(
                color_options[
                    math.floor((v.item() - most_neg) * 99 / color_range)
                ].hex_l
            )

    labels = [f"{x:.4f}" for x in activations]

    Xn, Yn = [], []
    xs_space = np.linspace(0.01, 2.99, len(layer_node_count))
    for layer_num, n_rows in enumerate(layer_node_count):
        Xn += [xs_space[layer_num]] * n_rows
        Yn += list(np.linspace(1.99, 0.01, n_rows))

    Xe, Ye = [], []
    for edge in edges:
        Xe += [Xn[edge[0]], Xn[edge[1]], None]
        Ye += [Yn[edge[0]], Yn[edge[1]], None]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=Xe,
            y=Ye,
            mode="lines",
            marker=dict(
                color="rgba(0, 0, 255, 0.2)",
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=Xn,
            y=Yn,
            mode="markers",
            text=labels,
            hoverinfo="text",
            opacity=0.7,
            marker=dict(
                size=10,
                color=colors,
            ),
        )
    )
    return fig


class ToyModel(nn.Module):
    def __init__(self, neurons=5, features=20, act_fn="ReLU"):
        super().__init__()
        self.neurons = neurons
        self.features = features
        self.W = nn.Parameter(torch.randn(neurons, features))
        self.b = nn.Parameter(torch.randn(features))
        self.act_fn = act_fn
        self.layer_cache = []

    def forward(self, x):
        self.layer_cache.append(x)
        x = einops.einsum(self.W, x, "inner outer, batch outer -> batch inner")
        self.layer_cache.append(x)
        x = einops.einsum(self.W.T, x, "outer inner, batch inner -> batch outer")
        self.layer_cache.append(x)
        x = x + self.b
        self.layer_cache.append(x)
        x = act_fn(self.act_fn)(x)
        self.layer_cache.append(x)
        return x, 0

    @staticmethod
    def plot(train_result):
        w = train_result.model.W.detach().numpy()
        px.imshow((w.T @ w)).show()
        px.imshow(w).show()

    def activations_graph(self, inputs):
        self.layer_cache = []
        self.forward(inputs)
        return activations_graph(self.layer_cache)

    def plots(self):
        w = self.W.cpu().detach().numpy()
        return dict(
            w_square=px.imshow((w.T @ w), title="w.T @ w"),
            w=px.imshow(w, title="w"),
            b=px.bar(self.b.cpu().detach().numpy(), title="b"),
            weights_sankey=weights_sankey([w, w.T]),
            # activations_sankey(
            #     [w],
            #     np.array([input_0, input_1, input_2]),
            # ),
        )


class LayerNorm(nn.Module):
    def __init__(self, length, eps=1e-5):
        super().__init__()
        self.length = length
        self.eps = eps
        self.w = nn.Parameter(torch.ones(self.length))
        self.b = nn.Parameter(torch.zeros(self.length))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x - x.mean(-1, keepdim=True)  # [batch, length]
        scale = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        x = x / scale  # [batch, length]
        return x * self.w + self.b


class LayerNormToyModel(nn.Module):
    def __init__(self, neurons=5, features=20, act_fn="SoLU"):
        super().__init__()
        self.neurons = neurons
        self.features = features
        self.W = nn.Parameter(torch.randn(neurons, features))
        self.b = nn.Parameter(torch.randn(features))
        self.act_fn = act_fn
        self.ln = LayerNorm(features)

    def forward(self, x):
        x = einops.einsum(self.W, x, "inner outer, batch outer -> batch inner")
        x = einops.einsum(self.W.T, x, "outer inner, batch inner -> batch outer")
        x = x + self.b
        x = act_fn(self.act_fn)(x)
        x = self.ln(x)
        return x, 0

    @staticmethod
    def plot(train_result):
        model = train_result.model
        ln = model.ln
        w = model.W.detach().numpy()
        px.imshow((w.T @ w), title="w.T @ w").show()
        px.imshow(w).show()
        px.bar(ln.w.detach().numpy(), title="LN W").show()
        px.bar(ln.b.detach().numpy(), title="LN B").show()

    def plots(self):
        w = self.W.detach().numpy()
        return dict(
            w_square=px.imshow((w.T @ w), title="w.T @ w"),
            w=px.imshow(w, title="w"),
            b=px.bar(self.b.detach().numpy(), title="b"),
            ln_w=px.bar(self.ln.w.detach().numpy(), title="LN W"),
            ln_b=px.bar(self.ln.b.detach().numpy(), title="LN B"),
        )


class HiddenLayerModel(nn.Module):
    def __init__(self, neurons, features, act_fn="ReLU"):
        super().__init__()
        self.neurons = neurons
        self.features = features
        self.W = nn.Parameter(torch.randn(neurons, features))
        self.b = nn.Parameter(torch.randn(features))
        self.act_fn = act_fn

    def forward(self, x):
        f = act_fn(self.act_fn)
        h = f(einops.einsum(self.W, x, "inner outer, batch outer -> batch inner"))
        x_ = f(
            einops.einsum(self.W.T, h, "outer inner, batch inner -> batch outer")
            + self.b
        )
        return x_, h

    @staticmethod
    def plot(train_result):
        w = train_result.model.W.detach().numpy()
        px.imshow(w.T).show()


class HiddenLayerModelVariation(nn.Module):
    def __init__(self, neurons=6, features=3, act_fn="ReLU"):
        super().__init__()
        self.neurons = neurons
        self.features = features
        self.W1 = nn.Parameter(torch.randn(neurons, features))
        self.W2 = nn.Parameter(torch.randn(features, neurons))
        self.b = nn.Parameter(torch.randn(features))
        self.act_fn = act_fn
        self.layer_cache = []

    def cache(self, layer):
        self.layer_cache.append(layer)
        return layer

    def forward(self, x):
        f = act_fn(self.act_fn)
        self.cache(x)
        x = self.cache(
            einops.einsum(self.W1, x, "inner outer, batch outer -> batch inner")
        )
        x = self.cache(f(x))
        x = self.cache(
            einops.einsum(self.W2, x, "outer inner, batch inner -> batch outer")
        )
        x = self.cache(x + self.b)
        x = self.cache(f(x))
        return x, 0

    @staticmethod
    def plots(train_result):
        model = train_result.model
        w1 = model.W1.detach().numpy()
        w2 = model.W2.detach().numpy()

        return (
            px.imshow(w1.T, title="W1"),
            stack_plot(w1),
            px.imshow(w2, title="W2"),
            stack_plot(w2.T),
            px.bar(model.b.detach().numpy(), title="bias"),
        )


class MLP(nn.Module):
    def __init__(self, d_model=5, d_mlp=20, act_fn="ReLU"):
        super().__init__()
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp))
        self.b_in = nn.Parameter(torch.randn(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model))
        self.b_out = nn.Parameter(torch.randn(d_model))
        self.act_fn = act_fn

    def forward(self, x):
        x = einops.einsum(self.W_in, x, "d_model d_mlp, batch d_model -> batch d_mlp")
        x = x + self.b_in
        x = act_fn(self.act_fn)(x)
        x = einops.einsum(self.W_out, x, "d_mlp d_model, batch d_mlp -> batch d_model")
        x = x + self.b_out
        return x


class MlpModel(nn.Module):
    def __init__(self, act_fn, features=20, d_model=5, d_mlp=20):
        super().__init__()
        self.features = features
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.W_E = nn.Parameter(torch.randn(features, d_model))
        self.mlp = MLP(d_model, d_mlp, act_fn)
        self.W_U = nn.Parameter(torch.randn(d_model, features))
        self.act_fn = act_fn

    def forward(self, x):
        x = einops.einsum(
            self.W_E, x, "features d_model, batch features -> batch d_model"
        )
        x = self.mlp(x)
        x = einops.einsum(
            self.W_U, x, "d_model features, batch d_model -> batch features"
        )
        return x, 0

    # TODO: plot MLP
    @staticmethod
    def plot(train_result):
        model = train_result.model
        w_e = model.W_E.detach().numpy()
        w_u = model.W_U.detach().numpy()
        px.imshow(w_e, title="W_E").show()
        stack_plot(w_e.T).show()
        px.imshow(w_u.T, title="W_U").show()
        stack_plot(w_u).show()


class ResidualModel(nn.Module):
    def __init__(self, features=20, d_model=5, d_mlp=20, act_fn="ReLU"):
        super().__init__()
        self.features = features
        self.d_model = d_model
        self.d_mlp = d_mlp
        self.W_E = nn.Parameter(torch.randn(features, d_model))
        self.mlp = MLP(d_model, d_mlp, act_fn)
        self.W_U = nn.Parameter(torch.randn(d_model, features))
        self.act_fn = act_fn

    def forward(self, x):
        x = einops.einsum(
            self.W_E, x, "features d_model, batch features -> batch d_model"
        )
        x = x + self.mlp(x)
        x = einops.einsum(
            self.W_U, x, "d_model features, batch d_model -> batch features"
        )
        return x, 0

    # TODO: plot MLP
    @staticmethod
    def plot(train_result):
        model = train_result.model
        w_e = model.W_E.detach().numpy()
        w_u = model.W_U.detach().numpy()
        px.imshow(w_e, title="W_E").show()
        stack_plot(w_e.T).show()
        px.imshow(w_u.T, title="W_U").show()
        stack_plot(w_u).show()
