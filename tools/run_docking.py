#!/usr/bin/env python3
"""AMD-BioForge -- Virtual Screening (Docking).

Usage:
    python scripts/run_docking.py --receptor target.pdb --ligands library.sdf
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from bioforge.models.docking_net import DockingNet
from bioforge.utils.rocm_utils import get_amd_gpu_info


def main():
    parser = argparse.ArgumentParser(description="AMD-BioForge: Molecular Docking")
    parser.add_argument("--receptor", type=str, required=True, help="Receptor PDB file")
    parser.add_argument("--ligands", type=str, required=True, help="Ligand library (SDF)")
    parser.add_argument("--exhaustiveness", type=int, default=8, help="Search exhaustiveness")
    parser.add_argument("--num-poses", type=int, default=10, help="Number of poses per ligand")
    parser.add_argument("--output", type=str, default="docking_results.csv")
    parser.add_argument("--predict-affinity", action="store_true", help="Use ML affinity predictor")
    args = parser.parse_args()

    gpu = get_amd_gpu_info()
    if not gpu["detected"]:
        print("AMD GPU not detected. Docking will be slow.")

    print("""
============================================================
    AMD-BioForge -- Virtual Screening
============================================================
""")

    print(f"Receptor: {args.receptor}")
    print(f"Ligands:  {args.ligands}")
    print(f"Exhaustiveness: {args.exhaustiveness}")
    print(f"Output:   {args.output}")

    if gpu["detected"]:
        print(f"GPU: {gpu['name']} ({gpu['vram_gb']} GB)")
        if gpu.get("fp64_tflops"):
            print(f"FP64: {gpu['fp64_tflops']} TFLOPS")

    # TODO: Implement full docking pipeline
    print("\n[info] Docking pipeline stub -- full implementation coming in v0.4.0")
    print("[info] See ROADMAP.md for development timeline")


if __name__ == "__main__":
    main()
