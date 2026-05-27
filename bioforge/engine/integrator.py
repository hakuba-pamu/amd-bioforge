"""Molecular Dynamics Integrators.

Velocity Verlet with Langevin thermostat.
FP64 throughout for energy conservation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class LangevinIntegrator(nn.Module):
    """Velocity Verlet integrator with Langevin thermostat.

    dx/dt = v
    dv/dt = F/m - gamma*v + sqrt(2*gamma*kT/m)*R(t)
    """

    def __init__(
        self,
        dt: float = 0.002,
        temperature: float = 300.0,
        friction: float = 1.0,
        masses: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.dt = dt
        self.temperature = temperature
        self.friction = friction

        self.kB = 0.001987204  # kcal/mol/K

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.masses: Optional[torch.Tensor] = None
        self._current_velocities: Optional[torch.Tensor] = None
        self.N = 0

        if masses is not None:
            self.set_masses(masses)

    def set_masses(self, masses: torch.Tensor) -> None:
        """Set particle masses for integration."""
        self.masses = masses.to(self.device, dtype=torch.float64)
        self._inv_masses = 1.0 / self.masses
        self.N = len(masses)

    def step(
        self,
        positions: torch.Tensor,
        velocities: torch.Tensor,
        forces: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Single Velocity Verlet integration step with Langevin thermostat."""
        if self.masses is None:
            raise ValueError("Masses not set. Call set_masses() first.")

        self._current_velocities = velocities

        kT = self.kB * self.temperature
        noise_scale = torch.sqrt(2.0 * self.friction * kT * self.dt)

        noise = torch.randn_like(velocities) * noise_scale * torch.sqrt(self._inv_masses).unsqueeze(-1)

        velocities = velocities + 0.5 * self.dt * (
            forces * self._inv_masses.unsqueeze(-1)
            - self.friction * velocities
            + noise
        )

        positions = positions + self.dt * velocities

        return positions, velocities

    @property
    def kinetic_energy(self) -> float:
        if self._current_velocities is None:
            return 0.0
        v2 = (self._current_velocities ** 2).sum()
        return 0.5 * (self.masses * v2).sum().item()

    @property
    def temperature_current(self) -> float:
        if self._current_velocities is None or self.N == 0:
            return 0.0
        dof = 3 * self.N - 3
        if dof <= 0:
            return 0.0
        return 2.0 * self.kinetic_energy / (self.kB * dof)
