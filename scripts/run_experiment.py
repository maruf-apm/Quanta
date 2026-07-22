#!/usr/bin/env python3
"""
Run a single experiment (ANN or PINN) for a given test case.
Usage:
    python scripts/run_experiment.py --config configs/tests/test1.yaml --model_type pinn
"""
import argparse
import sys
import os

# Allow imports from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
import numpy as np
from utils.config import load_config
from utils.logger import setup_logger
from physics.equations import EQUATION_REGISTRY
from models.networks import FeedForwardNN
from models.pinn_solver import PINNSolver
from models.ann_solver import ANNSolver
from data.dataset import ODEDataGenerator
from training.trainer import Trainer
from evaluation.plotter import ODEPlotter
from evaluation.table_exporter import TableExporter
from evaluation.metrics import Metrics


def main():
    parser = argparse.ArgumentParser(description="Quanta1: Single ODE Experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to test config YAML")
    parser.add_argument("--model_type", type=str, choices=["ann", "pinn"], required=True)
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--n_eval", type=int, default=100, help="Evaluation grid points")
    args = parser.parse_args()

    logger = setup_logger()
    cfg = load_config(args.config)
    test_name = cfg["name"]
    out_dir = os.path.join(args.output_dir, test_name, args.model_type)
    os.makedirs(out_dir, exist_ok=True)

    # Reproducibility
    seed = cfg["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Equation
    eq_cls = EQUATION_REGISTRY[cfg["equation"]]
    equation = eq_cls()
    logger.info(f"Loaded equation: {equation.name} | Domain: {equation.domain}")

    # Network
    mcfg = cfg.get("model", {})
    network = FeedForwardNN(
        input_dim=mcfg.get("input_dim", 1),
        output_dim=mcfg.get("output_dim", 1),
        hidden_layers=mcfg.get("hidden_layers", [50, 50, 50, 50]),
        activation=mcfg.get("activation", "tanh"),
    )

    # Solver
    if args.model_type == "pinn":
        model = PINNSolver(network, equation)
    else:
        model = ANNSolver(network)

    # Data
    data_gen = ODEDataGenerator(
        equation=equation,
        n_collocation=cfg["training"].get("n_collocation", 100),
        seed=seed,
    )

    # Train
    trainer = Trainer(
        model=model,
        config={
            "epochs": cfg["training"]["epochs"],
            "learning_rate": cfg["training"].get("learning_rate", 1e-4),
            "device": args.device,
            "save_dir": out_dir,
        },
    )
    logger.info(f"Starting {args.model_type.upper()} training for {test_name}...")
    history = trainer.train(data_gen, model_type=args.model_type)
    logger.info("Training complete.")

    # Evaluate
    n_test, m_real = data_gen.generate_test_grid(n_points=args.n_eval)
    model.eval()
    with torch.no_grad():
        m_pred = model(torch.tensor(n_test, dtype=torch.float32)).numpy()

    metrics = {
        "MAE": Metrics.mae(m_real, m_pred),
        "RMSE": Metrics.rmse(m_real, m_pred),
        "MaxError": Metrics.max_abs_error(m_real, m_pred),
    }
    logger.info(f"Metrics: {metrics}")

    # Plotting
    plotter = ODEPlotter(output_dir=os.path.join(out_dir, "figures"))
    plotter.plot_training_loss(
        history,
        title=f"{args.model_type.upper()} Training Loss - {test_name}",
        color=cfg["plotting"].get("loss_color", "blue"),
        filename="training_loss.png",
    )

    # Export table
    exporter = TableExporter(output_dir=os.path.join(out_dir, "tables"))
    exporter.export_single_prediction(n_test, m_real, m_pred, test_name, args.model_type)
    exporter.export_metrics(metrics, test_name, args.model_type)

    # Checkpoint
    trainer.save_checkpoint(f"{args.model_type}_model.pt")
    logger.info(f"All artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
