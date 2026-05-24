"""Unit tests for MD integrator."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioforge.engine.integrator import LangevinIntegrator


class TestLangevinIntegrator:
    def test_initialization(self):
        masses = torch.tensor([12.0, 14.0, 16.0], dtype=torch.float64)
        integrator = LangevinIntegrator(
            dt=0.002,
            temperature=300.0,
            friction=1.0,
            masses=masses,
        )
        assert integrator.dt == 0.002
        assert integrator.temperature == 300.0
        assert integrator.N == 3

    def test_set_masses(self):
        integrator = LangevinIntegrator()
        masses = torch.tensor([12.0, 14.0], dtype=torch.float64)
        integrator.set_masses(masses)
        assert integrator.N == 2

    def test_step_without_masses(self):
        integrator = LangevinIntegrator()
        pos = torch.zeros(2, 3)
        vel = torch.zeros(2, 3)
        forces = torch.zeros(2, 3)
        with pytest.raises(ValueError):
            integrator.step(pos, vel, forces)

    def test_temperature(self):
        masses = torch.tensor([12.0] * 10, dtype=torch.float64)
        integrator = LangevinIntegrator(dt=0.002, temperature=300.0, masses=masses)

        # Random velocities at ~300K
        kT = 0.001987204 * 300.0
        vel = torch.randn(10, 3, dtype=torch.float64) * torch.sqrt(kT / masses).unsqueeze(-1)
        pos = torch.zeros(10, 3, dtype=torch.float64)
        forces = torch.zeros(10, 3, dtype=torch.float64)

        integrator.step(pos, vel, forces)
        temp = integrator.temperature_current

        # Temperature should be roughly in the right ballpark
        assert 100 < temp < 600  # Wide range due to small system
