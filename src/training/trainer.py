"""
Unified training engine for both ANN and PINN models.
"""
import os
import torch
from torch.optim import Adam
from tqdm import tqdm
from typing import Dict, Any


class Trainer:
    """
    Generic trainer that adapts to ANN or PINN based on model type.

    Parameters
    ----------
    model : BaseODESolver
        The solver instance (ANN or PINN).
    config : dict
        Training hyperparameters.
    """

    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device(config.get("device", "cpu"))
        self.model.to(self.device)

        lr = config.get("learning_rate", 1e-4)
        self.optimizer = Adam(self.model.parameters(), lr=lr)
        self.epochs = config.get("epochs", 5000)
        self.save_dir = config.get("save_dir", "./results")
        os.makedirs(self.save_dir, exist_ok=True)

        self.history = []

    def train(self, data_generator, model_type: str = "pinn"):
        """
        Execute training loop.

        Parameters
        ----------
        data_generator : ODEDataGenerator
        model_type : str
            'pinn' or 'ann'.
        """
        self.model.train()

        # Prepare data
        n_collocation = data_generator.generate_collocation_points().to(self.device)
        n_ic, m_ic = data_generator.generate_initial_condition()
        n_ic, m_ic = n_ic.to(self.device), m_ic.to(self.device)

        if model_type == "ann":
            n_exact, m_exact = data_generator.generate_exact_points()
            n_exact, m_exact = n_exact.to(self.device), m_exact.to(self.device)

        pbar = tqdm(range(self.epochs), desc=f"Training {model_type.upper()}")
        for epoch in pbar:
            self.optimizer.zero_grad()

            if model_type == "pinn":
                losses = self.model.compute_loss(n_collocation, n_ic, m_ic)
            else:
                losses = self.model.compute_loss(n_exact, m_exact)

            loss = losses["total"]
            loss.backward()
            self.optimizer.step()

            # Record history
            record = {"epoch": epoch}
            for k, v in losses.items():
                record[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
            self.history.append(record)

            if epoch % 100 == 0:
                pbar.set_postfix({k: f"{v:.2e}" for k, v in record.items() if k != "epoch"})

        return self.history

    def save_checkpoint(self, filename: str = "model.pt"):
        """Save model weights, optimizer state, and history."""
        path = os.path.join(self.save_dir, filename)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "history": self.history,
                "config": self.config,
            },
            path,
        )
        return path
