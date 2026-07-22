"""
Export solution comparison tables and metrics to CSV.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict


class TableExporter:
    """Handles CSV export for reproducible result tables."""

    def __init__(self, output_dir: str = "./results/tables"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def export_comparison(self, n_values: np.ndarray, m_real: np.ndarray,
                          m_pinn: np.ndarray, m_ann: np.ndarray, test_name: str):
        """Save side-by-side solution comparison (like Tables 1, 5, 9, ...)."""
        data = {
            "n": n_values.flatten(),
            "Real Solution": m_real.flatten(),
            "PINN Solution": m_pinn.flatten(),
            "ANN Solution": m_ann.flatten(),
            "Real-PINN": np.abs(m_real - m_pinn).flatten(),
            "Real-ANN": np.abs(m_real - m_ann).flatten(),
        }
        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, f"{test_name}_comparison.csv")
        df.to_csv(path, index=False, float_format="%.10f")
        return path

    def export_metrics(self, metrics_dict: Dict[str, float], test_name: str,
                       model_type: str = "model"):
        """Save MAE/RMSE metrics (like Tables 3, 4, 7, 8, ...)."""
        df = pd.DataFrame([metrics_dict])
        path = os.path.join(self.output_dir, f"{test_name}_{model_type}_metrics.csv")
        df.to_csv(path, index=False)
        return path

    def export_single_prediction(self, n_values: np.ndarray, m_real: np.ndarray,
                                 m_pred: np.ndarray, test_name: str, model_type: str):
        """Export single-model prediction vs real."""
        data = {
            "n": n_values.flatten(),
            "Real Solution": m_real.flatten(),
            f"{model_type.upper()} Solution": m_pred.flatten(),
            f"Real-{model_type.upper()}": np.abs(m_real - m_pred).flatten(),
        }
        df = pd.DataFrame(data)
        path = os.path.join(self.output_dir, f"{test_name}_{model_type}_comparison.csv")
        df.to_csv(path, index=False, float_format="%.10f")
        return path
