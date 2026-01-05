# Research Project Template

This is a GitHub template repository for starting new research projects that use the [smon-toolbox](https://github.com/chipnbits/smon-toolbox).

## Using This Template

### On GitHub

1. Click "Use this template" button on GitHub
2. Create a new repository from this template
3. Clone your new repository
4. Follow the setup instructions below

### Locally

1. Copy this template directory to your desired location:
   ```bash
   cp -r templates/project-template ../my-new-project
   cd ../my-new-project
   rm TEMPLATE_README.md  # Remove this file
   ```

2. Update the following files:
   - `README.md`: Replace placeholder text with your project details
   - `pyproject.toml`: Update project name, description, and author info
   - `.github/workflows/*`: Customize CI/CD if needed

3. Initialize git and make first commit:
   ```bash
   git init
   git add .
   git commit -m "Initial commit from template"
   ```

## Setup Instructions

After creating a project from this template:

1. Install `uv`:
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Set up the environment:
   ```bash
   uv sync
   ```

3. Install pre-commit hooks:
   ```bash
   uv run pre-commit install
   ```

4. Start working!

## What's Included

- ✅ **Modern dependency management** with `uv`
- ✅ **Pre-configured smon-toolbox** as a git dependency
- ✅ **Pre-commit hooks** for code formatting (black, isort)
- ✅ **Project structure** for reproducible ML experiments
- ✅ **Example files** for experiments, scripts, and notebooks
- ✅ **Comprehensive .gitignore** for Python/ML projects
- ✅ **Best practices README** template

## Project Structure

```
.
├── notebooks/              # Jupyter notebooks
├── scripts/                # Utility scripts
├── configs/                # Configuration files
├── experiments/            # Experiment code
│   └── example_experiment/
├── data/                   # Data (gitignored)
│   ├── raw/
│   ├── processed/
│   └── external/
├── figures/                # Generated figures
├── pyproject.toml         # Dependencies
├── .pre-commit-config.yaml
└── README.md
```

## Local Development with smon-toolbox

For active development where you're editing both the toolbox and your project:

```bash
# Clone smon-toolbox
git clone https://github.com/chipnbits/smon-toolbox.git ../smon-toolbox

# Install in editable mode
uv add --editable ../smon-toolbox/packages/smon-toolbox
```

Now changes to smon-toolbox are instantly available in your project!

## Next Steps

1. Delete this `TEMPLATE_README.md` file
2. Update `README.md` with your project details
3. Customize the example experiment
4. Start your research!
