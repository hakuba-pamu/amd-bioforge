# AMD-BioForge

GPU-accelerated molecular dynamics and protein-ligand docking, optimized for AMD hardware.

This is a research tool I built for my bioinformatics work at ITB. Most MD software is heavily NVIDIA-optimized; I wanted something that takes advantage of AMD's FP64 throughput on CDNA cards.

## Components

- **md_engine** — Velocity Verlet integrator with HIP-accelerated force fields (AMBER, CHARMM)
- **docking** — Rigid docking with grid-based scoring, GPU-native
- **affinity** — Binding affinity prediction using a fine-tuned ESM-2 variant
- **analysis** — RMSD, RMSF, hydrogen bonds, radius of gyration

## Performance

On MI210 (CDNA2), comparing to GROMACS on A100:

| Workload | BioForge/MI210 | GROMACS/A100 | Notes |
|----------|---------------|-------------|-------|
| 50k atom NPT | 12.3 ns/day | 18.7 ns/day | ~66% of A100, not bad |
| 100k atom NVE | 8.1 ns/day | 14.2 ns/day | FP64 heavy |
| Docking (1000 ligands) | 340s | 290s | Scoring function differs |

These are rough numbers from my own benchmarks. Your mileage will vary.

## Requirements

- AMD GPU with ROCm 6.1+ (CDNA recommended, RDNA3 partially works)
- Python 3.10+
- See `requirements.txt` for full deps

```bash
pip install -e .
```

## Citation

If you use this in your research, please cite:

```bibtex
@software{bioforge2025,
  author = {Pamungkas, R.},
  title = {AMD-BioForge: GPU-Accelerated Computational Biology for AMD},
  year = {2025},
  url = {https://github.com/hakuba-pamu/amd-bioforge}
}
```

## Known issues

- RDNA3 cards have FP64 performance issues with large systems (>200k atoms)
- The docking module is slow compared to AutoDock-GPU — working on it
- PBC handling has edge cases with non-orthogonal boxes

MIT License.


## Troubleshooting
**Q: Getting OOM errors?**
A: Reduce batch size or enable gradient checkpointing.