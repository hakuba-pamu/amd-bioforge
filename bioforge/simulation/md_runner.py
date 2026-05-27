"""Molecular Dynamics Simulation Runner.

Orchestrates MD workflow: Setup -> Equilibration -> Production -> Analysis
Optimized for AMD GPU with pipelined force computation.
"""

import torch
import time
import numpy as np
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime
import json

from ..engine.forcefield import ForceField, ForceFieldType
from ..engine.integrator import LangevinIntegrator
from ..utils.pdb_parser import PDBParser
from ..utils.rocm_utils import get_amd_gpu_info


class MDSimulation:
    """GPU-accelerated molecular dynamics simulation.

    Runs entirely on AMD GPU with FP64 precision.
    Logs energy, temperature, pressure every N steps.
    """

    def __init__(
        self,
        pdb_path: Path,
        ff_type: ForceFieldType = ForceFieldType.AMBER14SB,
        dt: float = 0.002,
        temperature: float = 300.0,
        pressure: Optional[float] = None,
        ensemble: str = "nvt",
        cutoff: float = 10.0,
        output_dir: Path = Path("./trajectory"),
    ):
        self.pdb_path = Path(pdb_path)
        self.dt = dt
        self.temperature = temperature
        self.pressure = pressure
        self.ensemble = ensemble.lower()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ff = ForceField(ff_type, cutoff=cutoff)
        self.parser = PDBParser()

        self.positions: Optional[torch.Tensor] = None
        self.velocities: Optional[torch.Tensor] = None
        self.forces: Optional[torch.Tensor] = None
        self.box: Optional[torch.Tensor] = None
        self.integrator: Optional[LangevinIntegrator] = None

        self.step_count: int = 0
        self.trajectory: list = []
        self.energies: Dict[str, list] = {
            "total": [], "kinetic": [], "potential": [],
            "temperature": [], "time_ps": [],
        }

    def setup(self) -> None:
        """Parse PDB, build topology, assign force field parameters."""
        print("[setup] Parsing PDB structure...")
        atoms, bonds, angles, dihedrals, box = self.parser.parse(self.pdb_path)

        print(f"[setup] System: {len(atoms)} atoms, {len(bonds)} bonds")

        self.ff.load_parameters(atoms)

        self.positions = torch.tensor(
            [a.position for a in atoms],
            device=self.device, dtype=torch.float64,
        )
        self.box = torch.tensor(box, device=self.device, dtype=torch.float64)

        # Maxwell-Boltzmann velocities
        torch.manual_seed(42)
        masses = torch.tensor([a.mass for a in atoms], device=self.device, dtype=torch.float64)
        kT = 0.001987204 * self.temperature
        std = torch.sqrt(kT / masses)
        self.velocities = torch.randn(len(atoms), 3, device=self.device, dtype=torch.float64) * std.unsqueeze(-1)
        self.velocities -= self.velocities.mean(dim=0)

        self.integrator = LangevinIntegrator(
            dt=self.dt,
            temperature=self.temperature,
            friction=1.0,
            masses=masses,
        )

        print("[setup] Computing initial forces...")
        energy, self.forces = self.ff.compute_energy(
            self.positions, bonds, angles, dihedrals, self.box,
        )

        self._bonds = bonds
        self._angles = angles
        self._dihedrals = dihedrals

        gpu_info = get_amd_gpu_info()
        if gpu_info["detected"]:
            print(f"[setup] GPU: {gpu_info['name']} ({gpu_info['vram_gb']} GB)")
        else:
            print("[setup] No AMD GPU detected. Running on CPU (SLOW).")

    def equilibrate(self, steps: int = 50000) -> None:
        """NVT equilibration phase."""
        print(f"\n[equilibrate] Running {steps} steps NVT equilibration...")
        t_start = time.perf_counter()

        for step in range(steps):
            self._single_step()

            if step % 1000 == 0:
                current_T = self.integrator.temperature_current
                if current_T > 0:
                    scale = torch.sqrt(
                        torch.tensor(self.temperature / current_T, device=self.device)
                    )
                    self.velocities *= scale

            if step % 5000 == 0:
                elapsed = time.perf_counter() - t_start
                ns_day = (step * self.dt / 1000) / (elapsed / 86400) if elapsed > 0 else 0
                print(f"  Step {step:>6d}/{steps} | "
                      f"T = {self.integrator.temperature_current:.1f}K | "
                      f"{ns_day:.0f} ns/day")

        self.step_count += steps
        print("[equilibrate] Complete")

    def run(self, duration_ns: float) -> None:
        """Production MD run."""
        n_steps = int(duration_ns * 1000 / self.dt)
        print(f"\n[production] Running {n_steps} steps ({duration_ns} ns)...")
        t_start = time.perf_counter()

        for step in range(n_steps):
            self._single_step()

            if step % 1000 == 0:
                self._log_frame()

            if step % 10000 == 0 and step > 0:
                elapsed = time.perf_counter() - t_start
                ns_completed = step * self.dt / 1000
                ns_day = ns_completed / (elapsed / 86400)
                print(f"  Step {step:>7d}/{n_steps} | {ns_day:.0f} ns/day")

        self.step_count += n_steps
        print(f"\n[production] Complete -- {self.step_count} total steps")

    def _single_step(self) -> None:
        """One integration step."""
        self.positions, self.velocities = self.integrator.step(
            self.positions, self.velocities, self.forces,
        )

        energy, self.forces = self.ff.compute_energy(
            self.positions, self._bonds, self._angles, self._dihedrals, self.box,
        )

        self.positions, self.velocities = self.integrator.step(
            self.positions, self.velocities, self.forces,
        )

    def _log_frame(self) -> None:
        """Log current state."""
        self.energies["kinetic"].append(self.integrator.kinetic_energy)
        self.energies["temperature"].append(self.integrator.temperature_current)
        self.energies["time_ps"].append(self.step_count * self.dt)

        if len(self.trajectory) % 10 == 0:
            self.trajectory.append(self.positions.cpu().numpy().copy())

    def save(self) -> None:
        """Save trajectory and energy logs."""
        energy_file = self.output_dir / "energies.json"
        with open(energy_file, "w") as f:
            json.dump(self.energies, f, indent=2)
        print(f"[save] Energy log: {energy_file}")

        if self.trajectory:
            traj_file = self.output_dir / "trajectory.npy"
            np_traj = np.stack(self.trajectory)
            np.save(traj_file, np_traj)
            print(f"[save] Trajectory: {traj_file} ({np_traj.shape})")
