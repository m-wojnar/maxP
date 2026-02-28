"""Parametrized MLP — muP (Maximal Update Parametrization)."""

import torch
import torch.nn as nn
from maxp_new import ParametrizedModule, Parametrization


class ParametrizedMLP(nn.Module):
    """MLP where every Linear is wrapped with ParametrizedModule.

    Same architecture as the vanilla MLP, but each layer declares its
    width_dim and layer_type so that Parametrization can set (a,b,c).
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

        # Input layer: fan-in is hidden_dim (output side scales with width)
        layers = [
            ParametrizedModule(
                nn.Linear(input_dim, hidden_dim, bias=False),
                width_dim=hidden_dim,
                layer_type="embedding",
            )
        ]

        # Hidden layers
        for _ in range(n_layers - 2):
            layers.append(
                ParametrizedModule(
                    nn.Linear(hidden_dim, hidden_dim, bias=False),
                    width_dim=hidden_dim,
                    layer_type="hidden",
                )
            )

        # Output layer: fan-in is hidden_dim, output is fixed
        layers.append(
            ParametrizedModule(
                nn.Linear(hidden_dim, output_dim, bias=False),
                width_dim=hidden_dim,
                layer_type="readout",
            )
        )

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x).relu()
        return self.layers[-1](x)


def make_parametrized_mlp(
    hidden_dim: int = 512,
    n_layers: int = 4,
    lr_prefactor: float = 0.01,
    optimizer_type: str = "adam",
    input_dim: int = 3072,
    output_dim: int = 10,
) -> tuple[ParametrizedMLP, Parametrization]:
    """Build a ParametrizedMLP and apply muP parametrization.

    Returns the model and the Parametrization object (use .param_groups
    for the optimizer).
    """
    model = ParametrizedMLP(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        output_dim=output_dim,
    )
    param = Parametrization(
        model,
        optimizer_type=optimizer_type,
        alignment="full",
        lr_prefactor=lr_prefactor,
    )
    return model, param
