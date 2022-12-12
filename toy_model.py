import plotly.express as px
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import einops
import tqdm.auto as tqdm
import pandas
from typing import Callable, List
import plotly.graph_objects as go
from numpy.typing import NDArray
from data import TrainConfig, TrainResult, ActFn


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
def loss_fn(I, y_pred, y_true):
    _, features = y_pred.shape
    error = y_true - y_pred
    # importance falls off geometrically
    importance = torch.tensor([I**i for i in range(features)])
    return torch.mean(error**2 * importance)


def train_model(config: TrainConfig):
    model = create_model(config)
    features: int = model.features

    lower_bound, upper_bound = -10, 10

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
    x_train = x_train[(x_train == 0).sum(axis=1) != features]
    x_test = x_test[(x_test == 0).sum(axis=1) != features]

    # Train model
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.98))

    task = lambda x: x
    if config.task == "SQUARE":
        task = lambda x: x**2
    elif config.task == "ABS":
        task = lambda x: abs(x)

    losses = []
    for t in tqdm.tqdm(range(config.steps)):
        prediction, l1_terms = model(x_train)
        actual = task(x_train)
        loss = (
            loss_fn(config.i, prediction, actual)
            + config.regularization_coeff * l1_terms.abs().sum()
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


def sankey(title: str, sources, targets, values, node_names, layer_shapes):
    colors, labels, xs, ys = [], [], [], []

    for i in range(len(values)):
        colors.append("rgba(255,0,0, 0.3)" if values[i] > 0 else "rgba(0,0,255, 0.3)")
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

    return sankey("Weights", sources, targets, values, node_names, layer_shapes)


class ToyModel(nn.Module):
    def __init__(self, neurons=5, features=20, act_fn="ReLU"):
        super().__init__()
        self.neurons = neurons
        self.features = features
        self.W = nn.Parameter(torch.randn(neurons, features))
        self.b = nn.Parameter(torch.randn(features))
        self.act_fn = act_fn

    def forward(self, x):
        x = einops.einsum(self.W, x, "inner outer, batch outer -> batch inner")
        x = einops.einsum(self.W.T, x, "outer inner, batch inner -> batch outer")
        x = x + self.b
        x = act_fn(self.act_fn)(x)
        return x, 0

    @staticmethod
    def plot(train_result):
        w = train_result.model.W.detach().numpy()
        px.imshow((w.T @ w)).show()
        px.imshow(w).show()


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

    def forward(self, x):
        f = act_fn(self.act_fn)
        x = einops.einsum(self.W1, x, "inner outer, batch outer -> batch inner")
        x = f(x)
        x = einops.einsum(self.W2, x, "outer inner, batch inner -> batch outer")
        x = x + self.b
        x = f(x)
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
