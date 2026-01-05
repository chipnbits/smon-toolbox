# PhD Computational Toolbox
This repository contains a comprehensive toolbox and set of scripts developed during my PhD research at UBC. It is structured as a large repository to allow code resuse across multiple projects, experiments, and languages.

## Installation
We use [uv](https://github.com/astral-sh/uv) for dependency management and workspace handling. It relies on a unified lockfile (`uv.lock`) to ensure reproducible environments across all machines. To set up a local development environment, simply sync the repository. This will create a local virtual environment (`.venv`) and install all workspace packages and dependencies exactly as specified in `uv.lock`.
following command:
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh
```

### Python Packages
**Option 1**: For global conda usage it can be installed by create a new conda environment, then installing via uv in editable mode:
```bash
conda create -n smon_toolbox python=3.13 -y
conda activate smon_toolbox
# Install the specific toolbox package
uv pip install -e packages/smon-toolbox
```
**Option 2**: For local dev usage, you can directly install the package in editable mode using uv to manage the venv:
```bash
uv sync --all-packages
```

### Structure
The root is structured as a monorepo containing multiple packages and scripts, while the inner packages are structured as standard Python packages. Each package contains its own `README.md` file with specific instructions and documentation.
