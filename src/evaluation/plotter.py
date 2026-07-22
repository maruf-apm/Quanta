"""
Visualization utilities replicating figures from Audu et al. (2026).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


class ODEPlotter:
    """
    Generates publication-quality figures:
      - Training loss curves (log-scale)
      - 2x3 heatmap comparisons (Real / PINN / Error / Real / ANN / Error)
      - Collocation point distributions
    """

    def __init__(self, output_dir: str = "./results/figures"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def plot_training_loss(self, history, title: str = "Training Loss",
                           color: str = "blue", filename: str = "training_loss.png"):
        """Replicate Figure 4, 6, 8, 10, 12, 14 style."""
        fig, ax = plt.subplots(figsize=(8, 6))
        epochs = [h["epoch"] for h in history]
        losses = [h["total"] for h in history]
        ax.semilogy(epochs, losses, color=color, linewidth=1.5, label="Training loss")
        ax.set_xlabel("Epochs", fontsize=13)
        ax.set_ylabel("Loss", fontsize=13)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_heatmaps(self, n_grid: np.ndarray, m_real: np.ndarray,
                      m_pinn: np.ndarray, m_ann: np.ndarray,
                      test_name: str, cmap: str = "coolwarm"):
        """
        Replicate paper heatmap layout: 2 rows x 3 columns.
        Top: Real | PINN | PINN Error
        Bottom: Real | ANN (Audu et al. 2024) | ANN Error
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        N = len(n_grid)
        # Replicate 1D solution along dummy y-axis to create 2D image
        m_real_2d = np.tile(m_real.reshape(1, -1), (N, 1))
        m_pinn_2d = np.tile(m_pinn.reshape(1, -1), (N, 1))
        m_ann_2d = np.tile(m_ann.reshape(1, -1), (N, 1))

        err_pinn = np.abs(m_real - m_pinn)
        err_ann = np.abs(m_real - m_ann)
        err_pinn_2d = np.tile(err_pinn.reshape(1, -1), (N, 1))
        err_ann_2d = np.tile(err_ann.reshape(1, -1), (N, 1))

        n_min, n_max = float(n_grid.min()), float(n_grid.max())

        # Determine color scale for errors
        vmax_pinn = np.max(err_pinn) * 1.1 if np.max(err_pinn) > 0 else 1e-10
        vmax_ann = np.max(err_ann) * 1.1 if np.max(err_ann) > 0 else 1e-10

        configs = [
            (m_real_2d, "Real Solution", None, False),
            (m_pinn_2d, "PINN_FODE Solution", None, False),
            (err_pinn_2d, f"Error: {np.max(err_pinn):.2e}", r"$\times 10^{-3}$", True),
            (m_real_2d, "Real Solution", None, False),
            (m_ann_2d, "ANN Solution (Audu et al. 2024)", None, False),
            (err_ann_2d, f"Error: {np.max(err_ann):.2e}", r"$\times 10^{-3}$", True),
        ]

        for ax, (data, title, cbar_label, is_error) in zip(axes.flat, configs):
            if is_error:
                im = ax.imshow(data, aspect="auto", origin="lower",
                               extent=[n_min, n_max, 0, 1],
                               cmap="RdBu_r", vmin=-vmax_pinn if "PINN" in title else -vmax_ann,
                               vmax=vmax_pinn if "PINN" in title else vmax_ann)
            else:
                im = ax.imshow(data, aspect="auto", origin="lower",
                               extent=[n_min, n_max, 0, 1],
                               cmap=cmap)
            ax.set_xlabel("n", fontsize=11)
            ax.set_ylabel("m", fontsize=11)
            ax.set_title(title, fontsize=12)
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if cbar_label:
                cbar.set_label(cbar_label, rotation=270, labelpad=18, fontsize=10)
            else:
                cbar.set_label("E", rotation=270, labelpad=18, fontsize=10)

        plt.tight_layout()
        path = os.path.join(self.output_dir, f"{test_name}_heatmap.png")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_collocation_points(self, n_collocation: np.ndarray,
                                n_ic: np.ndarray, filename: str = "collocation_points.png"):
        """Replicate Figure 16: collocation and boundary point distribution."""
        fig, ax = plt.subplots(figsize=(10, 8))
        m_collocation = np.random.uniform(0, 1, size=n_collocation.shape)
        m_ic = np.zeros_like(n_ic)

        ax.scatter(n_collocation, m_collocation, s=6, c="blue", alpha=0.5,
                   label="Collocation Points", zorder=3)
        ax.scatter(n_ic, m_ic, s=50, c="red", marker="*",
                   label="Initial condition", zorder=4)
        ax.axvline(0, color="magenta", linewidth=2.5, label="Boundary condition at m = 0")
        ax.axvline(1, color="magenta", linewidth=2.5, label="Boundary condition at m = 1")
        ax.axhline(0, color="magenta", linewidth=2.5)
        ax.axhline(1, color="magenta", linewidth=2.5)

        ax.set_xlabel("n", fontsize=13)
        ax.set_ylabel("m", fontsize=13)
        ax.set_title("Collocation Points Distribution", fontsize=14)
        ax.legend(loc="upper right", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        path = os.path.join(self.output_dir, filename)
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return path
