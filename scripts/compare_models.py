#!/usr/bin/env python3
"""
Run both ANN and PINN on a test case and generate comparison figures/tables.
Replicates the side-by-side analysis from Audu et al. (2026).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from data.dataset import ODEDataGenerator
from evaluation.metrics import Metrics
from evaluation.plotter import ODEPlotter
from evaluation.table_exporter import TableExporter
from models.ann_solver import ANNSolver
from models.networks import FeedForwardNN
from models.pinn_solver import PINNSolver
from physics.equations import EQUATION_REGISTRY
from training.trainer import Trainer
from utils.config import load_config
from utils.logger import setup_logger


def train_model(model, model_type, data_gen, cfg, device, out_dir):
    trainer = Trainer(
        model=model,
        config={
            "epochs": cfg["training"]["epochs"],
            "learning_rate": cfg["training"].get("learning_rate", 1e-4),
            "device": device,
            "save_dir": out_dir,
        },
    )
    history = trainer.train(data_gen, model_type=model_type)
    return trainer, history


def main():
    parser = argparse.ArgumentParser(description="Quanta1: ANN vs PINN Comparison")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_eval", type=int, default=100)
    args = parser.parse_args()

    logger = setup_logger()
    cfg = load_config(args.config)
    test_name = cfg["name"]
    out_dir = os.path.join(args.output_dir, test_name, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    eq_cls = EQUATION_REGISTRY[cfg["equation"]]
    equation = eq_cls()
    logger.info(f"Comparing ANN vs PINN for {test_name}")

    data_gen = ODEDataGenerator(
        equation=equation,
        n_collocation=cfg["training"].get("n_collocation", 100),
        seed=seed,
    )

    mcfg = cfg.get("model", {})
    hidden = mcfg.get("hidden_layers", [50, 50, 50, 50])
    act = mcfg.get("activation", "tanh")

    # Train PINN
    net_pinn = FeedForwardNN(hidden_layers=hidden, activation=act)
    model_pinn = PINNSolver(net_pinn, equation)
    logger.info("Training PINN...")
    trainer_pinn, hist_pinn = train_model(
        model_pinn, "pinn", data_gen, cfg, args.device, os.path.join(out_dir, "pinn")
    )

    # Train ANN
    net_ann = FeedForwardNN(hidden_layers=hidden, activation=act)
    model_ann = ANNSolver(net_ann)
    logger.info("Training ANN...")
    trainer_ann, hist_ann = train_model(
        model_ann, "ann", data_gen, cfg, args.device, os.path.join(out_dir, "ann")
    )

    # Evaluation grid
    n_test, m_real = data_gen.generate_test_grid(n_points=args.n_eval)

    model_pinn.eval()
    model_ann.eval()
    with torch.no_grad():
        m_pinn = model_pinn(torch.tensor(n_test, dtype=torch.float32)).numpy()
        m_ann = model_ann(torch.tensor(n_test, dtype=torch.float32)).numpy()

    metrics_pinn = {
        "MAE": Metrics.mae(m_real, m_pinn),
        "RMSE": Metrics.rmse(m_real, m_pinn),
        "MaxError": Metrics.max_abs_error(m_real, m_pinn),
    }
    metrics_ann = {
        "MAE": Metrics.mae(m_real, m_ann),
        "RMSE": Metrics.rmse(m_real, m_ann),
        "MaxError": Metrics.max_abs_error(m_real, m_ann),
    }
    logger.info(f"PINN Metrics: {metrics_pinn}")
    logger.info(f"ANN  Metrics: {metrics_ann}")

    # Plots
    plotter = ODEPlotter(output_dir=os.path.join(out_dir, "figures"))
    plotter.plot_training_loss(
        hist_pinn,
        title=f"PINN Training Loss - {test_name}",
        color=cfg["plotting"].get("loss_color", "blue"),
        filename="pinn_training_loss.png",
    )
    plotter.plot_training_loss(
        hist_ann,
        title=f"ANN Training Loss - {test_name}",
        color="red",
        filename="ann_training_loss.png",
    )
    plotter.plot_heatmaps(
        n_test,
        m_real,
        m_pinn,
        m_ann,
        test_name,
        cmap=cfg["plotting"].get("colormap", "coolwarm"),
    )

    # Collocation points figure (once per test)
    n_col = data_gen.generate_collocation_points().numpy()
    n_ic, _ = data_gen.generate_initial_condition()
    plotter.plot_collocation_points(
        n_col, n_ic.numpy(), filename=f"{test_name}_collocation.png"
    )

    # Tables
    exporter = TableExporter(output_dir=os.path.join(out_dir, "tables"))
    exporter.export_comparison(n_test, m_real, m_pinn, m_ann, test_name)
    exporter.export_metrics(metrics_pinn, test_name, "pinn")
    exporter.export_metrics(metrics_ann, test_name, "ann")

    # Save checkpoints
    trainer_pinn.save_checkpoint("pinn_model.pt")
    trainer_ann.save_checkpoint("ann_model.pt")

    logger.info(f"Comparison complete. Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
