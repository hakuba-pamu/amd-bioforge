#!/usr/bin/env python3
"""AMD-BioForge -- GPU Benchmark Suite.

Benchmarks force computation, integration, and docking across system sizes.
"""

import sys
import time
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioforge.engine.forcefield import ForceField, ForceFieldType, AtomParams
from bioforge.engine.integrator import LangevinIntegrator
from bioforge.utils.rocm_utils import get_amd_gpu_info


def benchmark_force_computation(n_atoms_list: list = None):
    """Benchmark force computation across system sizes."""
    if n_atoms_list is None:
        n_atoms_list = [100, 500, 1000, 5000, 10000]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"  AMD-BioForge -- Force Computation Benchmark")
    print(f"{'='*60}")
    print(f"  {'Atoms':<10} {'Time (ms)':>12} {'Steps/sec':>12} {'ns/day':>10}")
    print(f"  {'-'*46}")

    for n_atoms in n_atoms_list:
        ff = ForceField(ForceFieldType.AMBER14SB)
        atom_params = [AtomParams(mass=12.0, charge=0.0, sigma=3.5, epsilon=0.1, atom_type="C")] * n_atoms
        ff.load_parameters(atom_params)

        positions = torch.randn(n_atoms, 3, device=device, dtype=torch.float64) * 10
        bonds = [(i, i+1) for i in range(n_atoms - 1)]
        box = torch.eye(3, device=device, dtype=torch.float64) * 100

        # Warmup
        for _ in range(5):
            _, _ = ff.compute_energy(positions, bonds, [], [], box)

        if device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            _, _ = ff.compute_energy(positions, bonds, [], [], box)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)

        avg_ms = np.mean(times)
        steps_sec = 1000.0 / avg_ms
        dt = 0.002  # ps
        ns_day = steps_sec * dt / 1000 * 86400

        print(f"  {n_atoms:<10} {avg_ms:>10.2f}ms {steps_sec:>10.0f} {ns_day:>8.0f}")

    print(f"{'='*60}")


def main():
    gpu = get_amd_gpu_info()

    print(f"""
============================================================
    AMD-BioForge -- Benchmark Suite
============================================================
  GPU:      {gpu.get('name', 'Unknown')}
  VRAM:     {gpu.get('vram_gb', 0)} GB
  FP64:     {gpu.get('fp64_tflops', 'N/A')} TFLOPS
  Detected: {gpu.get('detected', False)}
============================================================
""")

    benchmark_force_computation()

    print("\nBenchmark complete.")
    print("Compare with NVIDIA results: see README.md benchmarks section.")


if __name__ == "__main__":
    main()
