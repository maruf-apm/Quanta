"""
Physics-Informed Neural Network (PINN) solver.
Embeds ODE residual and initial condition into the loss function.
"""
import torch
from .base_model import BaseODESolver


class PINNSolver(BaseODESolver):
    """
    PINN solver using automatic differentiation for dm/dn.

    Loss = MSE_residual + MSE_initial_condition
    """

    def __init__(self, network, equation):
        super().__init__(network)
        self.equation = equation

    def compute_derivative(self, n: torch.Tensor):
        """Compute m and dm/dn via autograd."""
        n = n.requires_grad_(True)
        m = self.network(n)
        dm_dn = torch.autograd.grad(
            outputs=m,
            inputs=n,
            grad_outputs=torch.ones_like(m),
            create_graph=True,
            retain_graph=True,
        )[0]
        return m, dm_dn

    def compute_loss(self, n_collocation, n_ic=None, m_ic=None):
        """
        Parameters
        ----------
        n_collocation : torch.Tensor
            Collocation points for residual evaluation.
        n_ic : torch.Tensor, optional
            Initial condition point(s).
        m_ic : torch.Tensor, optional
            Initial condition value(s).
        """
        # Residual loss
        m_pred, dm_dn = self.compute_derivative(n_collocation)
        residual = self.equation.residual(n_collocation, m_pred, dm_dn)
        loss_ode = torch.mean(residual ** 2)

        # Initial condition loss
        loss_ic = torch.tensor(0.0, device=n_collocation.device)
        if n_ic is not None and m_ic is not None:
            m_ic_pred = self.network(n_ic)
            loss_ic = torch.mean((m_ic_pred - m_ic) ** 2)

        return {
            "total": loss_ode + loss_ic,
            "ode": loss_ode,
            "ic": loss_ic,
        }
