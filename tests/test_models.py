"""
Unit tests for Quanta1 solvers.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from physics.equations import Test1Equation, Test4Equation
from models.networks import FeedForwardNN
from models.pinn_solver import PINNSolver
from models.ann_solver import ANNSolver


def test_network_forward():
    net = FeedForwardNN(input_dim=1, output_dim=1, hidden_layers=[10, 10], activation="tanh")
    x = torch.randn(5, 1)
    y = net(x)
    assert y.shape == (5, 1), f"Expected (5,1), got {y.shape}"
    print("[PASS] test_network_forward")


def test_pinn_residual():
    eq = Test1Equation()
    net = FeedForwardNN(hidden_layers=[20, 20], activation="tanh")
    pinn = PINNSolver(net, eq)
    n = torch.linspace(0, 1, 10).view(-1, 1)
    m, dm = pinn.compute_derivative(n)
    assert m.shape == (10, 1)
    assert dm.shape == (10, 1)
    losses = pinn.compute_loss(n, *eq.initial_condition())
    assert "total" in losses
    print("[PASS] test_pinn_residual")


def test_ann_loss():
    net = FeedForwardNN(hidden_layers=[20, 20], activation="tanh")
    ann = ANNSolver(net)
    n = torch.linspace(0, 1, 10).view(-1, 1)
    m_exact = torch.sin(n)
    losses = ann.compute_loss(n, m_exact)
    assert "total" in losses
    assert losses["total"].item() >= 0
    print("[PASS] test_ann_loss")


def test_exact_solution():
    eq = Test4Equation()
    n = np.array([0.0, 0.5, 1.0])
    m = eq.exact(n)
    expected = np.array([1.0, 0.8, 0.5])
    assert np.allclose(m, expected, atol=1e-6)
    print("[PASS] test_exact_solution")


if __name__ == "__main__":
    test_network_forward()
    test_pinn_residual()
    test_ann_loss()
    test_exact_solution()
    print("\nAll tests passed!")
