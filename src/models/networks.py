"""
Neural network architectures for ODE solving.
"""
import torch
import torch.nn as nn
from typing import List


class FeedForwardNN(nn.Module):
    """
    Configurable multi-layer perceptron (MLP).

    Parameters
    ----------
    input_dim : int
        Dimension of input (independent variable n).
    output_dim : int
        Dimension of output (dependent variable m).
    hidden_layers : List[int]
        List specifying the number of neurons per hidden layer.
    activation : str
        Non-linearity: 'tanh', 'relu', 'sigmoid', 'leaky_relu'.
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_layers: List[int] = None,
        activation: str = "tanh",
    ):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [50, 50, 50, 50]

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers

        act_map = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "leaky_relu": nn.LeakyReLU(),
        }
        if activation.lower() not in act_map:
            raise ValueError(f"Unsupported activation: {activation}")
        self.activation = act_map[activation.lower()]

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(self.activation)
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Xavier initialization for stable training
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
