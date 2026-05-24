"""3D protein visualization using py3Dmol and matplotlib."""

import numpy as np
from typing import Optional
from pathlib import Path


class ProteinViewer:
    """Visualize protein structures and MD trajectories.

    Supports static rendering (matplotlib) and interactive (py3Dmol).
    """

    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height

    def plot_rmsd(
        self,
        rmsd: np.ndarray,
        time_ps: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot RMSD over time."""
        import matplotlib.pyplot as plt

        if time_ps is None:
            time_ps = np.arange(len(rmsd))

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(time_ps, rmsd, linewidth=1.5, color="#ED1C24")
        ax.set_xlabel("Time (ps)")
        ax.set_ylabel("RMSD (A)")
        ax.set_title("RMSD Over Time")
        ax.grid(True, alpha=0.3)

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[viz] Saved RMSD plot: {save_path}")
        plt.close(fig)

    def plot_energy(
        self,
        energies: dict,
        save_path: Optional[Path] = None,
    ) -> None:
        """Plot energy components over time."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        time_ps = energies.get("time_ps", [])

        if energies.get("kinetic"):
            axes[0].plot(time_ps, energies["kinetic"], label="Kinetic", color="#3366cc")
        if energies.get("total"):
            axes[0].plot(time_ps, energies["total"], label="Total", color="#dc3912")
        axes[0].set_ylabel("Energy (kcal/mol)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        if energies.get("temperature"):
            axes[1].plot(time_ps, energies["temperature"], color="#ff9900")
            axes[1].axhline(y=310, color="gray", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Time (ps)")
        axes[1].set_ylabel("Temperature (K)")
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("MD Simulation Monitoring")

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[viz] Saved energy plot: {save_path}")
        plt.close(fig)

    def view_3d(self, pdb_path: str) -> Optional[object]:
        """Interactive 3D viewer using py3Dmol."""
        try:
            import py3Dmol

            with open(pdb_path, "r") as f:
                pdb_data = f.read()

            viewer = py3Dmol.view(width=self.width, height=self.height)
            viewer.addModel(pdb_data, "pdb")
            viewer.setStyle({"cartoon": {"color": "spectrum"}})
            viewer.zoomTo()
            return viewer
        except ImportError:
            print("[viz] py3Dmol not installed. pip install py3Dmol")
            return None
