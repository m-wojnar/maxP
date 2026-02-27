"""Vanilla MLP — Standard Parametrization (SP) baseline."""

import torch.nn as nn


class MLP(nn.Module):
    """Simple MLP with standard PyTorch init (kaiming_uniform_).

    Architecture: Linear → ReLU → ... → Linear (no bias).
    This is the SP baseline: flat LR, no output multipliers.
    """

    def __init__(
        self,
        input_dim: int = 3072,
        hidden_dim: int = 512,
        n_layers: int = 4,
        output_dim: int = 10,
    ):
        super().__init__()
        assert n_layers >= 2, "Need at least input + output layer"
        layers = [nn.Linear(input_dim, hidden_dim, bias=False)]
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
        layers.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)
