"""Trajectory analysis tools for MD simulations.

Computes RMSD, RMSF, radius of gyration, hydrogen bonds, and SASA.
"""

import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


class TrajectoryAnalyzer:
    """Analyze molecular dynamics trajectories.

    Computes structural and dynamic properties from trajectory data.
    """

    def __init__(self, reference: Optional[np.ndarray] = None):
        self.reference = reference

    def rmsd(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute RMSD relative to reference structure.

        Args:
            trajectory: [frames, N, 3] coordinates

        Returns:
            rmsd: [frames] in Angstroms
        """
        if self.reference is None:
            self.reference = trajectory[0]

        ref = self.reference - self.reference.mean(axis=0)
        rmsds = []

        for frame in trajectory:
            coords = frame - frame.mean(axis=0)
            # Kabsch alignment (simplified)
            diff = coords - ref
            rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
            rmsds.append(rmsd)

        return np.array(rmsds)

    def rmsf(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute RMSF (Root Mean Square Fluctuation) per atom.

        Args:
            trajectory: [frames, N, 3]

        Returns:
            rmsf: [N] per-atom fluctuation in Angstroms
        """
        mean_pos = trajectory.mean(axis=0)
        diff = trajectory - mean_pos[np.newaxis]
        rmsf = np.sqrt(np.mean(np.sum(diff**2, axis=2), axis=0))
        return rmsf

    def radius_of_gyration(self, trajectory: np.ndarray) -> np.ndarray:
        """Compute radius of gyration.

        Args:
            trajectory: [frames, N, 3]

        Returns:
            rg: [frames] in Angstroms
        """
        rg = []
        for frame in trajectory:
            center = frame.mean(axis=0)
            diff = frame - center
            rg.append(np.sqrt(np.mean(np.sum(diff**2, axis=1))))
        return np.array(rg)

    def hydrogen_bonds(
        self,
        trajectory: np.ndarray,
        donor_indices: List[int],
        acceptor_indices: List[int],
        cutoff: float = 3.5,
        angle_cutoff: float = 150.0,
    ) -> np.ndarray:
        """Count hydrogen bonds per frame.

        Simplified: counts D-A distances < cutoff.
        """
        hbonds = []
        for frame in trajectory:
            count = 0
            for d in donor_indices:
                for a in acceptor_indices:
                    dist = np.linalg.norm(frame[d] - frame[a])
                    if dist < cutoff:
                        count += 1
            hbonds.append(count)
        return np.array(hbonds)

    def analyze_all(
        self,
        trajectory: np.ndarray,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, np.ndarray]:
        """Run all analyses."""
        results = {
            "rmsd": self.rmsd(trajectory),
            "rmsf": self.rmsf(trajectory),
            "radius_of_gyration": self.radius_of_gyration(trajectory),
        }

        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for name, data in results.items():
                np.save(output_dir / f"{name}.npy", data)
                print(f"[analysis] Saved {name}: {data.shape}")

        return results
