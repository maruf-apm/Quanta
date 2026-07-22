# Quanta1: Physics-Informed Neural Networks for First-Order ODEs

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Quanta1** is a production-grade research framework that reproduces and extends the methodology of *Audu et al. (2026)* — solving first-order ordinary differential equations (ODEs) using both **Physics-Informed Neural Networks (PINNs)** and standard **Artificial Neural Networks (ANNs)**. The entire pipeline is configurable via YAML, modular, and designed for easy extension to new equations or architectures.

---

## Features

- **Six Benchmark Tests** — All numerical experiments from the paper (Tests 1–6) including the metal-rod cooling problem.
- **Modular Architecture** — Swap ODEs, network architectures, optimizers, or loss functions without touching the core training loop.
- **YAML-Driven Configuration** — Every hyperparameter, from hidden-layer sizes to collocation-point counts, lives in a config file.
- **Exact Figure Reproduction** — Training-loss curves (log-scale), 2×3 heatmap comparisons, and collocation-point scatter plots match the paper’s style.
- **CSV Export** — Solution comparison tables and MAE/RMSE metrics exported automatically for LaTeX/paper integration.
- **Automatic Differentiation** — PINN residuals computed via PyTorch autograd; no finite-difference approximations.

---

## Repository Structure

```
Quanta1/
├── configs/
│   ├── model/
│   │   └── default.yaml          # Default network architecture
│   ├── training/
│   │   └── default.yaml          # Default optimizer & hyperparameters
│   └── tests/
│       ├── test1.yaml            # Test 1: dm/dn = n² - 4m
│       ├── test2.yaml            # Test 2: dm/dn = e^{-2n} - 5m
│       ├── test3.yaml            # Test 3: dm/dn = 1.5m - n³m
│       ├── test4.yaml            # Test 4: dm/dn = -2nm/(1+n²)
│       ├── test5.yaml            # Test 5: dm/dn = -0.5(m-3)
│       └── test6.yaml            # Test 6: Metal rod cooling (Stefan-Boltzmann)
├── src/
│   ├── physics/
│   │   └── equations.py          # ODE definitions + exact solutions
│   ├── models/
│   │   ├── networks.py           # FeedForwardNN builder
│   │   ├── base_model.py         # Abstract solver interface
│   │   ├── pinn_solver.py        # PINN with residual + IC loss
│   │   └── ann_solver.py         # Supervised ANN
│   ├── data/
│   │   └── dataset.py            # Collocation & exact-data generators
│   ├── training/
│   │   └── trainer.py            # Unified training engine
│   ├── evaluation/
│   │   ├── metrics.py            # MAE, RMSE, MaxError
│   │   ├── plotter.py            # Paper-quality figure generation
│   │   └── table_exporter.py     # CSV export utilities
│   └── utils/
│       ├── config.py             # YAML loader & merger
│       └── logger.py             # Standard logging setup
├── scripts/
│   ├── run_experiment.py         # Single-run entry point
│   ├── compare_models.py         # Side-by-side ANN vs PINN
│   └── generate_all_results.py   # Batch all 6 tests
├── tests/
│   └── test_models.py            # Unit tests for solvers
├── requirements.txt
├── setup.py
└── README.md
```

---

## Installation

```bash
git clone https://github.com/yourusername/Quanta1.git
cd Quanta1
pip install -r requirements.txt
# Optional: install as editable package
pip install -e .
```

---

## Quick Start

### 1. Run a single experiment (PINN on Test 1)

```bash
python scripts/run_experiment.py \
    --config configs/tests/test1.yaml \
    --model_type pinn \
    --output_dir ./results
```

### 2. Run a single experiment (ANN on Test 1)

```bash
python scripts/run_experiment.py \
    --config configs/tests/test1.yaml \
    --model_type ann \
    --output_dir ./results
```

### 3. Full side-by-side comparison (ANN vs PINN)

```bash
python scripts/compare_models.py \
    --config configs/tests/test1.yaml \
    --output_dir ./results
```

This produces:
- `results/test1/comparison/figures/training_loss.png`
- `results/test1/comparison/figures/test1_heatmap.png`
- `results/test1/comparison/tables/test1_comparison.csv`
- `results/test1/comparison/tables/test1_pinn_metrics.csv`
- `results/test1/comparison/tables/test1_ann_metrics.csv`

### 4. Batch all six tests

```bash
python scripts/generate_all_results.py
```

---

## Configuration Guide

Each test YAML file follows this schema:

```yaml
name: "test1"
equation: "Test1Equation"          # Class name from src/physics/equations.py
domain: [0.0, 1.0]

model:
  hidden_layers: [50, 50, 50, 50]  # 4 hidden layers, 50 neurons each
  activation: "tanh"               # tanh, relu, sigmoid, leaky_relu

training:
  epochs: 5000
  learning_rate: 0.0001
  n_collocation: 100               # Number of residual points
  seed: 42

plotting:
  colormap: "coolwarm"             # Matplotlib colormap for heatmaps
  loss_color: "blue"               # Training curve line color
```

To add a **new ODE**, simply:
1. Subclass `ODEEquation` in `src/physics/equations.py`.
2. Register it in `EQUATION_REGISTRY`.
3. Create a new YAML in `configs/tests/` pointing to your class.

No other files need modification.

---

## Reproducing Paper Figures

| Figure | Script Output |
|--------|---------------|
| Fig. 4, 6, 8, 10, 12, 14 (Training Loss) | `figures/*_training_loss.png` |
| Fig. 5, 7, 9, 11, 13, 15 (Heatmaps) | `figures/*_heatmap.png` |
| Fig. 16 (Collocation Points) | `figures/*_collocation.png` |
| Tables 1–24 | `tables/*.csv` |

---

## Citation

If you use Quanta1 in your research, please cite the original paper:

```bibtex
@article{audu2026pinn,
  title={Neural Network Solutions for First-Order Differential Equations: A Physics-Informed Perspective},
  author={Audu, K. J. and Kayode, A. O. and Fajuyi, S. O. and Inuolaji, M. O. and Yahaya, Y. A.},
  journal={Journal of Engineering and Basic Sciences},
  volume={5},
  pages={1836617},
  year={2026}
}
```

---

## License

MIT License — see `LICENSE` for details.
# Quanta
# Quanta
