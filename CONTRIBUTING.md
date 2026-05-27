# Contributing to AMD-BioForge

Contributions are welcome, particularly from researchers and developers working in computational biology or GPU computing.

## Areas where contributions are most useful

- Force field implementations (CHARMM36, OPLS-AA)
- Validation against GROMACS/NAMD on standard benchmarks
- RDNA3 FP64 optimization
- Additional analysis tools (PCA, free energy perturbation)
- Documentation and tutorials

## Development setup

```bash
git clone https://github.com/hakuba-pamu/amd-bioforge.git
cd amd-bioforge
pip install -e ".[dev]"
```

Run tests:
```bash
pytest test/ -v
```

## Submitting changes

- Open an issue first for large changes to discuss the approach
- Include benchmark results if your change affects performance
- Update relevant documentation
- Add tests for new functionality

## Code standards

- Type hints for public API functions
- Docstrings in Google style
- HIP kernels should include occupancy comments

I review PRs when I have time — usually within a week.
