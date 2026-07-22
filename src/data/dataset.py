"""
Data generation for collocation points, initial conditions, and exact solutions.
"""
import torch
import numpy as np
from typing import Tuple


class ODEDataGenerator:
    """
    Generates training and evaluation data for a given ODE equation.

    Parameters
    ----------
    equation : ODEEquation
        The ODE problem definition.
    n_collocation : int
        Number of collocation (residual) points.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, equation, n_collocation: int = 100, seed: int = 42):
        self.equation = equation
        self.n_collocation = n_collocation
        self.rng = np.random.default_rng(seed)

    def generate_collocation_points(self) -> torch.Tensor:
        """Uniform random collocation points in the domain."""
        n_min, n_max = self.equation.domain
        n = self.rng.uniform(n_min, n_max, (self.n_collocation, 1))
        return torch.tensor(n, dtype=torch.float32)

    def generate_initial_condition(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (n_ic, m_ic) tensors."""
        n0, m0 = self.equation.initial_condition()
        n_ic = torch.tensor([[n0]], dtype=torch.float32)
        m_ic = torch.tensor([[m0]], dtype=torch.float32)
        return n_ic, m_ic

    def generate_exact_points(self, n_points: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate (n, m_exact) pairs on a uniform grid."""
        if n_points is None:
            n_points = self.n_collocation
        n_min, n_max = self.equation.domain
        n = np.linspace(n_min, n_max, n_points).reshape(-1, 1)
        m = self.equation.exact(n)
        return torch.tensor(n, dtype=torch.float32), torch.tensor(m, dtype=torch.float32)

    def generate_test_grid(self, n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a fine uniform grid for evaluation."""
        n_min, n_max = self.equation.domain
        n = np.linspace(n_min, n_max, n_points).reshape(-1, 1)
        m = self.equation.exact(n)
        return n, m
