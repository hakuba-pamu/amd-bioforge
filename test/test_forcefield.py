"""Unit tests for force field computation."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioforge.engine.forcefield import ForceField, ForceFieldType, AtomParams


class TestForceField:
    def test_initialization(self):
        ff = ForceField(ForceFieldType.AMBER14SB)
        assert ff.ff_type == ForceFieldType.AMBER14SB
        assert ff.cutoff == 10.0

    def test_load_parameters(self):
        ff = ForceField()
        params = [
            AtomParams(mass=12.0, charge=0.0, sigma=3.5, epsilon=0.1, atom_type="C"),
            AtomParams(mass=14.0, charge=-0.5, sigma=3.2, epsilon=0.2, atom_type="N"),
        ]
        ff.load_parameters(params)
        assert ff.n_atoms == 2

    def test_bonded_energy(self):
        ff = ForceField()
        params = [AtomParams(mass=12.0, charge=0.0, sigma=3.5, epsilon=0.1, atom_type="C")] * 3
        ff.load_parameters(params)

        device = torch.device("cpu")
        positions = torch.tensor([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ], device=device, dtype=torch.float64)

        bonds = [(0, 1), (1, 2)]
        box = torch.eye(3, device=device, dtype=torch.float64) * 100

        energy, forces = ff.compute_energy(positions, bonds, [], [], box)
        assert energy.item() > 0  # Should have positive energy
        assert forces.shape == (3, 3)


class TestIntegrator:
    def test_velocity_verlet(self):
        from bioforge.engine.integrator import LangevinIntegrator

        masses = torch.tensor([12.0, 12.0], dtype=torch.float64)
        integrator = LangevinIntegrator(dt=0.002, temperature=300.0, masses=masses)

        pos = torch.zeros(2, 3, dtype=torch.float64)
        vel = torch.ones(2, 3, dtype=torch.float64)
        forces = torch.randn(2, 3, dtype=torch.float64)

        new_pos, new_vel = integrator.step(pos, vel, forces)

        assert new_pos.shape == (2, 3)
        assert new_vel.shape == (2, 3)
        assert not torch.allclose(new_pos, pos)  # Positions should change
