# `src/` — Core Library Documentation

This directory contains the modular source code for **Quanta1**, a Physics-Informed Neural Network (PINN) framework for solving first-order Ordinary Differential Equations (ODEs).

---

## Directory Structure

```
src/
├── physics/          # ODE definitions and exact solutions
├── models/           # Neural network architectures and solvers
├── data/             # Data generation utilities
├── training/         # Training engine and optimization
├── evaluation/       # Metrics, plotting, and export
└── utils/            # Configuration and logging helpers
```

---

## `physics/equations.py`

### Purpose
Defines all benchmark ODEs from Audu et al. (2026) with their exact analytical solutions.

### Key Classes

#### `ODEEquation` (Abstract Base)
```python
from physics.equations import ODEEquation

class MyCustomODE(ODEEquation):
    @property
    def name(self) -> str:
        return "MyCustomODE"

    def residual(self, n, m, dm_dn):
        """Return R = dm/dn - f(n,m). Should be ~0 for correct solution."""
        return dm_dn - (n ** 2 - m)

    def exact(self, n):
        """Analytical solution for validation."""
        return np.exp(n)  # or torch.exp(n)

    def initial_condition(self):
        """Return (n0, m0) tuple."""
        return (0.0, 1.0)

    @property
    def domain(self):
        """Computational domain (n_min, n_max)."""
        return (0.0, 1.0)
```

#### `EQUATION_REGISTRY`
Factory dictionary for YAML-driven instantiation:
```python
from physics.equations import EQUATION_REGISTRY

# Instantiate by string name from config
eq_cls = EQUATION_REGISTRY["Test1Equation"]
equation = eq_cls()
```

**Available equations:** `Test1Equation` through `Test6Equation`.

**Note on Test 3:** The paper text states `dm/dn = 1.5m - n³m` but the reported table values correspond to the sign-flipped ODE `dm/dn = n³m - 1.5m`. The implementation uses the latter to match the paper's numerical results.

**Note on Test 6:** Uses high-fidelity `scipy.integrate.solve_ivp` (RK45, `rtol=1e-10`) to generate the "exact" solution since no closed form exists.

---

## `models/networks.py`

### `FeedForwardNN`

Configurable multi-layer perceptron with Xavier initialization.

```python
from models.networks import FeedForwardNN

# Default: 4 hidden layers × 50 neurons, tanh activation
net = FeedForwardNN(
    input_dim=1,
    output_dim=1,
    hidden_layers=[50, 50, 50, 50],
    activation="tanh"
)

# Forward pass
import torch
n = torch.linspace(0, 1, 100).view(-1, 1)
m_pred = net(n)  # shape: (100, 1)
```

**Supported activations:** `tanh`, `relu`, `sigmoid`, `leaky_relu`.

---

## `models/base_model.py`

### `BaseODESolver`

Abstract base class enforcing the solver interface.

```python
from models.base_model import BaseODESolver

class MySolver(BaseODESolver):
    def compute_loss(self, **kwargs):
        # Must return dict with key "total"
        return {"total": loss_tensor, "custom": other_loss}
```

All solvers must implement `forward(n)` and `compute_loss(**kwargs)`.

---

## `models/pinn_solver.py`

### `PINNSolver`

Physics-Informed Neural Network solver. Embeds the ODE residual and initial condition into the loss function via automatic differentiation.

```python
from models.pinn_solver import PINNSolver
from physics.equations import Test1Equation
from models.networks import FeedForwardNN

# Build solver
equation = Test1Equation()
net = FeedForwardNN(hidden_layers=[50, 50, 50, 50])
model = PINNSolver(net, equation)

# Forward pass with IC-enforcing ansatz
n = torch.tensor([[0.0], [0.5], [1.0]], requires_grad=True)
m_pred = model(n)  # Uses m(n) = m0 + n·NN(n) for large-scale problems

# Compute derivative via autograd
m, dm_dn = model.compute_derivative(n)

# Compute total loss
n_collocation = torch.rand(100, 1)
n_ic = torch.tensor([[0.0]])
m_ic = torch.tensor([[1.0]])
losses = model.compute_loss(n_collocation, n_ic, m_ic)
# Returns: {"total": ..., "ode": ..., "ic": ...}
```

**Key design:** `forward()` applies an IC-enforcing ansatz `m(n) = m0 + n·NN(n)` which structurally guarantees `m(0) = m0`. This is critical for large-scale problems like Test 6 where `m0 = 1000` and raw tanh outputs `[-1, 1]` are insufficient.

---

## `models/ann_solver.py`

### `ANNSolver`

Standard supervised neural network trained on exact solution labels.

```python
from models.ann_solver import ANNSolver

model = ANNSolver(net)

# Requires exact labels
n = torch.linspace(0, 1, 100).view(-1, 1)
m_exact = equation.exact(n)  # or pre-computed labels
losses = model.compute_loss(n, m_exact)
# Returns: {"total": ..., "mse": ...}
```

**Use case:** Baseline comparison against PINN. Does not embed physical laws — learns purely from data.

---

## `data/dataset.py`

### `ODEDataGenerator`

Generates collocation points, initial conditions, and exact solutions.

```python
from data.dataset import ODEDataGenerator

data_gen = ODEDataGenerator(
    equation=equation,
    n_collocation=100,
    seed=42
)

# Uniform random collocation points in domain
n_col = data_gen.generate_collocation_points()  # torch.Tensor (100, 1)

# Initial condition
n_ic, m_ic = data_gen.generate_initial_condition()  # (1,1), (1,1)

# Exact solution on uniform grid (for ANN training)
n_exact, m_exact = data_gen.generate_exact_points(n_points=100)

# Fine grid for evaluation
n_test, m_real = data_gen.generate_test_grid(n_points=100)
```

---

## `training/trainer.py`

### `Trainer`

Unified training engine for both ANN and PINN models.

```python
from training.trainer import Trainer

trainer = Trainer(
    model=model,
    config={
        "epochs": 5000,
        "learning_rate": 1e-4,
        "device": "cpu",
        "save_dir": "./results",
        "scheduler": "plateau"  # or "step", "none"
    }
)

# Train
history = trainer.train(data_gen, model_type="pinn")
# history = [{"epoch": 0, "total": 1.2, "ode": 0.8, "ic": 0.4}, ...]

# Save checkpoint
trainer.save_checkpoint("model.pt")
# Saves: model weights, optimizer state, history, config
```

**Features:**
- Adam optimizer with optional `ReduceLROnPlateau` or `StepLR` scheduling
- Gradient clipping (`max_norm=1.0`) for training stability
- Progress bar with live loss display

---

## `evaluation/metrics.py`

### `Metrics`

Static utility class for error computation.

```python
from evaluation.metrics import Metrics

mae = Metrics.mae(y_true, y_pred)       # Mean Absolute Error
rmse = Metrics.rmse(y_true, y_pred)       # Root Mean Squared Error
max_err = Metrics.max_abs_error(y_true, y_pred)  # Maximum absolute error
```

---

## `evaluation/plotter.py`

### `ODEPlotter`

Generates publication-quality figures matching Audu et al. (2026).

```python
from evaluation.plotter import ODEPlotter

plotter = ODEPlotter(output_dir="./results/figures")

# 1. Training loss curve (log-scale)
plotter.plot_training_loss(
    history,
    title="PINN Training Loss - test1",
    color="blue",
    filename="training_loss.png"
)

# 2. 2×3 heatmap comparison (Real / PINN / Error / Real / ANN / Error)
plotter.plot_heatmaps(
    n_grid, m_real, m_pinn, m_ann,
    test_name="test1",
    cmap="coolwarm"
)

# 3. Collocation point distribution (Figure 16 style)
plotter.plot_collocation_points(
    n_collocation.numpy(),
    n_ic.numpy(),
    filename="collocation.png"
)
```

---

## `evaluation/table_exporter.py`

### `TableExporter`

Exports CSV files for LaTeX integration and result tables.

```python
from evaluation.table_exporter import TableExporter

exporter = TableExporter(output_dir="./results/tables")

# Side-by-side comparison (like Tables 1, 5, 9...)
exporter.export_comparison(
    n_values, m_real, m_pinn, m_ann,
    test_name="test1"
)
# Output: test1_comparison.csv

# Metrics summary (like Tables 3, 4, 7, 8...)
exporter.export_metrics(
    {"MAE": 1.2e-8, "RMSE": 1.5e-8, "MaxError": 2.8e-8},
    test_name="test1",
    model_type="pinn"
)
# Output: test1_pinn_metrics.csv
```

---

## `utils/config.py`

### Configuration Loader

```python
from utils.config import load_config, merge_configs, build_config

# Load single YAML
cfg = load_config("configs/tests/test1.yaml")

# Merge configs (test-specific overrides model defaults)
final_cfg = build_config(
    test_config_path="configs/tests/test1.yaml",
    model_config_path="configs/model/default.yaml",
    training_config_path="configs/training/default.yaml"
)
```

---

## `utils/logger.py`

### Logging Setup

```python
from utils.logger import setup_logger

logger = setup_logger(name="Quanta1", level=logging.INFO)
logger.info("Training started...")
# Output: [2026-07-24 01:07:00] Quanta1 - INFO - Training started...
```

---

## Adding a New Test Case

1. **Create equation class** in `physics/equations.py`:
```python
class Test7Equation(ODEEquation):
    name = "Test7Equation"
    def residual(self, n, m, dm_dn):
        return dm_dn - (-0.1 * m)
    def exact(self, n):
        return np.exp(-0.1 * n)
    def initial_condition(self):
        return (0.0, 1.0)
    @property
    def domain(self):
        return (0.0, 10.0)
```

2. **Register in `EQUATION_REGISTRY`**:
```python
EQUATION_REGISTRY = {
    # ... existing entries ...
    "Test7Equation": Test7Equation,
}
```

3. **Create config** `configs/tests/test7.yaml`:
```yaml
name: "test7"
equation: "Test7Equation"
domain: [0.0, 10.0]
model:
  hidden_layers: [40, 40, 40, 40]
training:
  epochs: 3000
  learning_rate: 0.0001
plotting:
  colormap: "viridis"
  loss_color: "green"
```

4. **Run**:
```bash
python scripts/compare_models.py --config configs/tests/test7.yaml
```

No other files need modification.
