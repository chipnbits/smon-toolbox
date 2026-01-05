# Development Guide

This guide provides detailed information for developing the smon_toolbox.

## Repository Structure

```
smon_toolbox/
├── packages/
│   ├── smon-toolbox/      # Core computational methods
│   └── template/          # Package template
├── data/                  # Shared test data
│   ├── datasets/
│   │   └── matrices/
│   └── benchmarks/
├── scripts/               # Utility scripts
│   ├── benchmarks/
│   ├── profiling/
│   └── testing/
├── templates/             # Project templates
│   └── project-template/  # GitHub template for new projects
├── conftest.py           # Workspace-level pytest fixtures
├── pyproject.toml        # Workspace configuration
└── uv.lock              # Unified lockfile
```

## Workspace Management with uv

This is a uv workspace with multiple packages. The workspace uses a unified lockfile (`uv.lock`) for reproducibility.

### Installing the workspace

```bash
# Install all packages listed in root pyproject.toml
uv sync
```

### Adding dependencies

```bash
# Add dependency to smon-toolbox
cd packages/smon-toolbox
uv add numpy

# Add dev dependency
uv add --dev pytest
```

## Testing

### Workspace-level test data

Test data is stored in `data/` and accessed via pytest fixtures in `conftest.py`:

```python
def test_my_function(matrices_dir):
    """matrices_dir is provided by conftest.py fixture"""
    A = np.load(matrices_dir / "spd_10x10.npy")
    # Test your function
```

### Running tests

```bash
# From workspace root - runs all package tests
uv run pytest
# Run specific package tests
uv run --package smon-toolbox pytest
```

## Adding a New Package

Use the template package as a starting point:

```bash
# Copy template
cp -r packages/template packages/my-new-package

# Update pyproject.toml in the new package
# - Change name, description
# - Add dependencies

# Sync workspace
uv sync --all-packages
```

## Hub-and-Spoke Model

The smon_toolbox is designed as a **hub** that individual research projects (spokes) depend on.

### For projects using the toolbox

**Option 1: Git dependency (recommended for sharing)**
```toml
[tool.uv.sources]
smon-toolbox = { git = "https://github.com/chipnbits/smon-toolbox.git", branch = "main" }
```

**Option 2: Local editable install (for active development)**
```bash
# In your project directory
uv add --editable ../smon-toolbox/packages/smon-toolbox
```

## Git LFS

Large binary files use Git LFS. After cloning, ensure LFS files are pulled:

```bash
git lfs install
git lfs pull
```

Tracked file types: `.npy`, `.npz`, `.h5`, `.hdf5`, `.mat`, `.pt`, `.pth`, `.ckpt`


## Best Practices

1. **Keep the toolbox lean**: Only add well-tested, reusable methods
2. **Document thoroughly**: Include mathematical background and examples
3. **Test comprehensively**: Unit tests with known solutions
5. **Profile performance**: Benchmark against baseline implementations

## Questions or Issues?

Contact: sghyseli@cs.ubc.ca
