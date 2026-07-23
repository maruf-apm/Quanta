# src/training/trainer.py
import os
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau  # <-- ADD THIS IMPORT
from tqdm import tqdm
from typing import Dict, Any


class Trainer:
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

        # ============================================
        # ADD SCHEDULER HERE (inside __init__)
        # ============================================
        scheduler_type = config.get("scheduler", "plateau")
        if scheduler_type == "plateau":
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.5, patience=200, threshold=1e-4
            )
        else:
            self.scheduler = None

    def train(self, data_generator, model_type: str = "pinn"):
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

            # Optional: gradient clipping (also helps stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            # ============================================
            # ADD SCHEDULER STEP HERE (inside train loop)
            # ============================================
            if self.scheduler is not None:
                self.scheduler.step(loss.item())

            # Record history
            record = {"epoch": epoch}
            for k, v in losses.items():
                record[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
            self.history.append(record)

            if epoch % 100 == 0:
                pbar.set_postfix(
                    {k: f"{v:.2e}" for k, v in record.items() if k != "epoch"}
                )

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
