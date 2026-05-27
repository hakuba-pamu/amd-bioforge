"""Energy minimization using steepest descent and conjugate gradient.

GPU-accelerated with FP64 precision for numerical stability.
"""

import torch
from typing import Tuple, Optional
from ..engine.forcefield import ForceField


class EnergyMinimizer:
    """Energy minimization for molecular systems.

    Supports steepest descent and conjugate gradient methods.
    """

    def __init__(
        self,
        force_field: ForceField,
        max_steps: int = 10000,
        tolerance: float = 1e-4,
        method: str = "sd",
    ):
        self.ff = force_field
        self.max_steps = max_steps
        self.tolerance = tolerance
        self.method = method

    def minimize(
        self,
        positions: torch.Tensor,
        bonds: list,
        angles: list,
        dihedrals: list,
        box: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """Run energy minimization.

        Returns:
            minimized_positions: [N, 3]
            final_energy: scalar (kcal/mol)
        """
        pos = positions.clone().detach().requires_grad_(True)
        device = pos.device

        step_size = 0.001  # Initial step size (A)
        prev_energy = float("inf")

        print(f"[minimize] Starting {self.method.upper()} minimization...")

        for step in range(self.max_steps):
            energy, forces = self.ff.compute_energy(
                pos, bonds, angles, dihedrals, box,
            )

            if self.method == "sd":
                # Steepest descent
                pos_new = pos + step_size * forces
            else:
                # Conjugate gradient (simplified)
                if step == 0:
                    direction = forces
                else:
                    gamma = torch.dot(forces.flatten(), forces.flatten()) / (
                        torch.dot(prev_forces.flatten(), prev_forces.flatten()) + 1e-10
                    )
                    direction = forces + gamma * prev_direction
                pos_new = pos + step_size * direction
                prev_direction = direction.clone()

            prev_forces = forces.clone()

            # Check convergence
            max_force = forces.abs().max().item()
            if max_force < self.tolerance:
                print(f"[minimize] Converged at step {step} (max_force={max_force:.2e})")
                return pos_new.detach(), energy.item()

            # Adaptive step size
            if energy.item() < prev_energy:
                step_size *= 1.2
            else:
                step_size *= 0.5

            prev_energy = energy.item()
            pos = pos_new.detach().requires_grad_(True)

            if step % 1000 == 0:
                print(f"  Step {step:>5d} | E = {energy.item():.2f} kcal/mol | F_max = {max_force:.4f}")

        print(f"[minimize] Reached max steps ({self.max_steps})")
        return pos.detach(), energy.item()
