#!/usr/bin/env python3
"""AMD-BioForge -- Molecular Dynamics Simulation Runner.

Usage:
    python scripts/run_md.py --pdb data/1ubq.pdb --duration 100
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioforge import MDSimulation
from bioforge.engine.forcefield import ForceFieldType
from bioforge.utils.rocm_utils import get_amd_gpu_info


def main():
    parser = argparse.ArgumentParser(description="AMD-BioForge: MD Simulation")
    parser.add_argument("--pdb", type=str, required=True, help="Input PDB file")
    parser.add_argument("--duration", type=float, default=10.0, help="Duration (ns)")
    parser.add_argument("--temperature", type=float, default=310.0, help="Temperature (K)")
    parser.add_argument("--ff", type=str, default="amber14sb", help="Force field")
    parser.add_argument("--output", type=str, default="./trajectory/")
    parser.add_argument("--equilibration", type=int, default=50000, help="Equilibration steps")
    args = parser.parse_args()

    gpu = get_amd_gpu_info()
    if not gpu["detected"]:
        print("AMD GPU not detected. Running on CPU (will be SLOW).")
        print("MD simulations require FP64. AMD GPUs recommended.")

    print("""
============================================================
    AMD-BioForge -- MD Simulation
============================================================
""")

    ff_type = ForceFieldType.AMBER14SB if args.ff == "amber14sb" else ForceFieldType.CHARMM36M

    sim = MDSimulation(
        pdb_path=Path(args.pdb),
        ff_type=ff_type,
        temperature=args.temperature,
        ensemble="nvt",
        output_dir=Path(args.output),
    )

    sim.setup()
    sim.equilibrate(steps=args.equilibration)
    sim.run(duration_ns=args.duration)
    sim.save()

    print(f"\nMD simulation complete!")
    print(f"   Output: {args.output}")
    print(f"   Total steps: {sim.step_count}")


if __name__ == "__main__":
    main()
