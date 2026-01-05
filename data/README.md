# Workspace Test Data

This directory contains **shared test datasets and benchmark problems** for the smon_toolbox packages.

## Directory Structure

```
data/
├── datasets/          # Small test datasets for unit/integration tests
│   ├── matrices/      # Test matrices (small size, various properties)
│   ├── timeseries/    # Test time series data
│   └── networks/      # Small test networks/graphs
│
└── benchmarks/        # Benchmark problems for performance testing
    ├── linear/        # Linear algebra benchmarks
    ├── ode/           # ODE solver benchmarks
    └── optimization/  # Optimization benchmarks
```

Available fixtures:
- `data_dir` - Root data directory
- `datasets_dir` - data/datasets

## Running Tests

From the workspace root:

```bash
# Run all tests across all packages
uv run pytest
# Run tests for a specific package
uv run --package smon-toolbox pytest
# Run with verbose output
uv run pytest -v
```

### Adding New Test Data

1. **Keep it small**: Minimal but representative datasets
2. **Document the source**: Add README or comments
3. **Use Git LFS** for files >1MB:
   ```bash
   git lfs track "data/benchmarks/*.mat"
   ```
4. **Provide example usage** in tests

## Example: Creating Test Matrices

```python
# data/datasets/matrices/generate_test_matrices.py
import numpy as np

# 10x10 symmetric positive definite matrix
A = np.random.RandomState(42).randn(10, 10)
A = A @ A.T + 10 * np.eye(10)
np.save('spd_10x10.npy', A)
```

## See Also
- `/conftest.py` - Workspace-level pytest fixtures
- `scripts/benchmarks/` - Benchmark running scripts
