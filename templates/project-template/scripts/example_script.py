"""
Example utility script.

This is a template for utility scripts that process data, run benchmarks, etc.
"""

import argparse
from pathlib import Path


def main(args):
    """Main function."""
    print(f"Running script with input: {args.input}")
    # Your script logic here


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example utility script")
    parser.add_argument("--input", type=str, help="Input file or parameter")
    parser.add_argument("--output", type=str, default="output", help="Output directory")

    args = parser.parse_args()
    main(args)
