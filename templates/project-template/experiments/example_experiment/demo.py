"""
Example experiment script.

Run with: uv run python experiments/example_experiment/run.py
"""

from pathlib import Path

import numpy as np

# TODO: Import from smon-toolbox
# from smon_toolbox.linear import cholesky_factorization


def main():
    """Main experiment function."""
    print("Running example experiment...")

    # Set random seed for reproducibility
    np.random.seed(42)

    # Your experiment code here
    # ...

    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()
