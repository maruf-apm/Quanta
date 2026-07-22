"""
Abstract base class for ODE solvers.
"""
from abc import abstractmethod
import torch.nn as nn


class BaseODESolver(nn.Module):
    """
    Base class for ANN and PINN ODE solvers.
    All solvers must implement `forward` and `compute_loss`.
    """

    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, n):
        """Predict m(n)."""
        return self.network(n)

    @abstractmethod
    def compute_loss(self, **kwargs):
        """Return dict of losses. Must contain key 'total'."""
        raise NotImplementedError
