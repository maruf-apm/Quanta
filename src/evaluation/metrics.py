"""
Evaluation metrics: MAE and RMSE.
"""
import numpy as np


class Metrics:
    """Static metric utilities."""

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean(np.abs(y_true - y_pred)))

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    @staticmethod
    def max_abs_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.max(np.abs(y_true - y_pred)))
