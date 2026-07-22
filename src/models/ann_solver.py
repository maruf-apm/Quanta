"""
Standard Artificial Neural Network (ANN) solver.
Supervised learning against exact solution.
"""
import torch
from .base_model import BaseODESolver


class ANNSolver(BaseODESolver):
    """
    ANN solver trained on exact solution labels.

    Loss = MSE(predicted, exact)
    """

    def __init__(self, network):
        super().__init__(network)

    def compute_loss(self, n, m_exact):
        """
        Parameters
        ----------
        n : torch.Tensor
            Input points.
        m_exact : torch.Tensor
            Exact solution labels.
        """
        m_pred = self.network(n)
        loss = torch.mean((m_pred - m_exact) ** 2)
        return {"total": loss, "mse": loss}
