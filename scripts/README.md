# `scripts/` — CLI Entry Points

This directory contains executable scripts for running experiments, comparing models, and batch-processing all benchmark tests.

---

## `run_experiment.py`

### Purpose
Run a **single experiment** (ANN or PINN) for one test case. Use this for quick iteration on a specific configuration.

### Usage

```bash
python scripts/run_experiment.py \
    --config configs/tests/test1.yaml \
    --model_type pinn \
    --output_dir ./results \
    --device cpu \
    --n_eval 100
```

### Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--config` | ✅ | — | Path to test YAML config |
| `--model_type` | ✅ | — | `ann` or `pinn` |
| `--output_dir` | ❌ | `./results` | Root directory for outputs |
| `--device` | ❌ | `cpu` | `cpu` or `cuda` |
| `--n_eval` | ❌ | `100` | Number of evaluation grid points |

### What It Does

```python
# Pseudocode of the script flow:
1. Load YAML config
2. Instantiate equation from EQUATION_REGISTRY
3. Build FeedForwardNN from config["model"]
4. Wrap in PINNSolver or ANNSolver
5. Generate collocation/exact data
6. Train with Trainer (Adam + optional scheduler)
7. Evaluate on fine grid
8. Compute MAE, RMSE, MaxError
9. Save:
   - figures/training_loss.png
   - tables/*_comparison.csv
   - tables/*_metrics.csv
   - model checkpoint (*.pt)
```

### Example Output Structure

```
results/
└── test1/
    └── pinn/
        ├── figures/
        │   └── training_loss.png
        ├── tables/
        │   ├── test1_pinn_comparison.csv
        │   └── test1_pinn_metrics.csv
        └── pinn_model.pt
```

### Code Snippet

```python
# From scripts/run_experiment.py

def main():
    # Parse CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["ann", "pinn"], required=True)
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    # Instantiate equation
    eq_cls = EQUATION_REGISTRY[cfg["equation"]]
    equation = eq_cls()

    # Build network
    network = FeedForwardNN(
        hidden_layers=cfg["model"]["hidden_layers"],
        activation=cfg["model"]["activation"]
    )

    # Build solver
    if args.model_type == "pinn":
        model = PINNSolver(network, equation)
    else:
        model = ANNSolver(network)

    # Train
    trainer = Trainer(model, config={
        "epochs": cfg["training"]["epochs"],
        "learning_rate": cfg["training"]["learning_rate"],
        "scheduler": cfg["training"].get("scheduler", "none"),
        "device": args.device,
        "save_dir": out_dir
    })
    history = trainer.train(data_gen, model_type=args.model_type)

    # Evaluate and save
    # ... (metrics, plots, tables, checkpoint)
```

---

## `compare_models.py`

### Purpose
Run **both ANN and PINN** on the same test case and generate side-by-side comparison figures and tables. This replicates the paper's comparative analysis.

### Usage

```bash
python scripts/compare_models.py \
    --config configs/tests/test1.yaml \
    --output_dir ./results \
    --device cpu \
    --n_eval 100
```

### What It Produces

| Artifact | Description |
|----------|-------------|
| `figures/pinn_training_loss.png` | PINN loss curve |
| `figures/ann_training_loss.png` | ANN loss curve |
| `figures/{test_name}_heatmap.png` | 2×3 comparison heatmap |
| `figures/{test_name}_collocation.png` | Collocation point scatter |
| `tables/{test_name}_comparison.csv` | Side-by-side solution table |
| `tables/{test_name}_pinn_metrics.csv` | PINN MAE/RMSE/MaxError |
| `tables/{test_name}_ann_metrics.csv` | ANN MAE/RMSE/MaxError |
| `pinn/pinn_model.pt` | PINN checkpoint |
| `ann/ann_model.pt` | ANN checkpoint |

### Heatmap Layout

Replicates Figures 5, 7, 9, 11, 13, 15 from the paper:

```
┌─────────────┬─────────────┬─────────────────┐
│ Real Sol    │ PINN Sol    │ PINN Error      │
├─────────────┼─────────────┼─────────────────┤
│ Real Sol    │ ANN Sol     │ ANN Error       │
└─────────────┴─────────────┴─────────────────┘
```

### Code Snippet

```python
# From scripts/compare_models.py

def train_model(model, model_type, data_gen, cfg, device, out_dir):
    """Helper: trains either ANN or PINN, returns trainer + history."""
    trainer = Trainer(model, config={
        "epochs": cfg["training"]["epochs"],
        "learning_rate": cfg["training"]["learning_rate"],
        "scheduler": cfg["training"].get("scheduler", "none"),
        "device": device,
        "save_dir": out_dir
    })
    history = trainer.train(data_gen, model_type=model_type)
    return trainer, history

def main():
    # ... load config, setup ...

    # Train PINN
    net_pinn = FeedForwardNN(hidden_layers=hidden, activation=act)
    model_pinn = PINNSolver(net_pinn, equation)
    trainer_pinn, hist_pinn = train_model(
        model_pinn, "pinn", data_gen, cfg, device,
        os.path.join(out_dir, "pinn")
    )

    # Train ANN
    net_ann = FeedForwardNN(hidden_layers=hidden, activation=act)
    model_ann = ANNSolver(net_ann)
    trainer_ann, hist_ann = train_model(
        model_ann, "ann", data_gen, cfg, device,
        os.path.join(out_dir, "ann")
    )

    # Evaluate both on same grid
    n_test, m_real = data_gen.generate_test_grid(n_points=args.n_eval)
    with torch.no_grad():
        m_pinn = model_pinn(torch.tensor(n_test, dtype=torch.float32)).numpy()
        m_ann = model_ann(torch.tensor(n_test, dtype=torch.float32)).numpy()

    # Generate all comparison artifacts
    plotter.plot_heatmaps(n_test, m_real, m_pinn, m_ann, test_name, cmap=...)
    exporter.export_comparison(n_test, m_real, m_pinn, m_ann, test_name)
    exporter.export_metrics(metrics_pinn, test_name, "pinn")
    exporter.export_metrics(metrics_ann, test_name, "ann")
```

---

## `generate_all_results.py`

### Purpose
**Batch script** that runs `compare_models.py` for all 6 test cases sequentially. Use this to regenerate the full result suite overnight.

### Usage

```bash
python scripts/generate_all_results.py
```

No arguments — it uses the hardcoded list of configs.

### What It Does

```python
# From scripts/generate_all_results.py

TEST_CONFIGS = [
    "configs/tests/test1.yaml",
    "configs/tests/test2.yaml",
    "configs/tests/test3.yaml",
    "configs/tests/test4.yaml",
    "configs/tests/test5.yaml",
    "configs/tests/test6.yaml",
]

def run_comparison(config_path):
    cmd = [
        sys.executable,
        "scripts/compare_models.py",
        "--config", config_path,
        "--output_dir", "./results",
    ]
    subprocess.run(cmd, check=True)

def main():
    for cfg in TEST_CONFIGS:
        if os.path.exists(cfg):
            print(f"\n{'='*60}")
            print(f"Running: {cfg}")
            print(f"{'='*60}")
            run_comparison(cfg)
        else:
            print(f"Config not found: {cfg}, skipping.")
    print("\nAll experiments completed.")
```

### Expected Runtime

| Test | Epochs | Approx. Time (CPU) |
|------|--------|-------------------|
| 1 | 5000 | ~2–5 min |
| 2 | 2000 | ~1–2 min |
| 3 | 2000 | ~1–2 min |
| 4 | 5000 | ~2–5 min |
| 5 | 2000 | ~1–2 min |
| 6 | 2000 | ~1–2 min |
| **Total** | — | **~10–20 min** |

### Full Output Structure

```
results/
├── test1/
│   └── comparison/
│       ├── pinn/
│       │   ├── figures/training_loss.png
│       │   └── tables/
│       ├── ann/
│       │   ├── figures/training_loss.png
│       │   └── tables/
│       ├── figures/
│       │   ├── pinn_training_loss.png
│       │   ├── ann_training_loss.png
│       │   ├── test1_heatmap.png
│       │   └── test1_collocation.png
│       └── tables/
│           ├── test1_comparison.csv
│           ├── test1_pinn_metrics.csv
│           └── test1_ann_metrics.csv
├── test2/
│   └── comparison/
│       └── ...
├── test3/
│   └── comparison/
│       └── ...
├── test4/
│   └── comparison/
│       └── ...
├── test5/
│   └── comparison/
│       └── ...
└── test6/
    └── comparison/
        └── ...
```

---

## Quick Reference: Which Script to Use?

| Goal | Script | Example |
|------|--------|---------|
| Quick PINN test on one case | `run_experiment.py` | `python scripts/run_experiment.py --config configs/tests/test3.yaml --model_type pinn` |
| Side-by-side ANN vs PINN | `compare_models.py` | `python scripts/compare_models.py --config configs/tests/test3.yaml` |
| Reproduce all paper results | `generate_all_results.py` | `python scripts/generate_all_results.py` |
| Debug a single equation | `run_experiment.py` | `python scripts/run_experiment.py --config configs/tests/test6.yaml --model_type pinn` |

---

## Adding a New Script

To add a custom analysis script:

```python
#!/usr/bin/env python3
"""
Custom analysis script template.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils.config import load_config
from physics.equations import EQUATION_REGISTRY
from models.networks import FeedForwardNN
from models.pinn_solver import PINNSolver
from data.dataset import ODEDataGenerator
# ... import what you need

def main():
    # Your analysis here
    pass

if __name__ == "__main__":
    main()
```

Always include the `sys.path.insert` block so imports resolve correctly from the `scripts/` directory.
