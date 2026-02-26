"""Minimal MLP training example using the new maxp Parametrization API.

Trains a simple MLP on synthetic regression data.
"""

import torch
import torch.nn as nn

import maxp_new as maxp


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, n_hidden: int = 2):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_in, d_hidden), nn.ReLU()]
        for _ in range(n_hidden - 1):
            layers += [nn.Linear(d_hidden, d_hidden), nn.ReLU()]
        layers.append(nn.Linear(d_hidden, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main():
    torch.manual_seed(42)

    # Synthetic regression data
    d_in, d_hidden, d_out = 32, 128, 1
    X = torch.randn(512, d_in)
    y = X[:, :3].sum(dim=1, keepdim=True) + 0.1 * torch.randn(512, 1)

    model = MLP(d_in, d_hidden, d_out, n_hidden=2)

    # muP a,b values: embedding(-0.5, 0.5), hidden(0.0, 0.5), readout(0.5, 0.5)
    # c is solved via LP with full alignment assumption
    param = maxp.Parametrization(
        model,
        layers={
            "net.0": {"a": -0.5, "b": 0.5},  # embedding
            "net.2": {"a": 0.0, "b": 0.5},   # hidden
            "net.4": {"a": 0.5, "b": 0.5},   # readout
        },
        optimizer_type="adam",
        alignment="full",
        lr_prefactor=1e-2,
    )
    optimizer = torch.optim.AdamW(param.param_groups)

    # Training loop
    for epoch in range(20):
        perm = torch.randperm(X.size(0))
        epoch_loss = 0.0
        n_batches = 0
        for i in range(0, X.size(0), 64):
            idx = perm[i : i + 64]
            xb, yb = X[idx], y[idx]

            loss = nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        if epoch % 5 == 0:
            print(f"Epoch {epoch:3d}  loss={avg_loss:.4f}")

    print(f"Final loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
