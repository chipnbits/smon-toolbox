"""Visualization utilities for smon-toolbox."""

from importlib.resources import files

import matplotlib.pyplot as plt

from .image_utils import (
    animate_batch,
    denormalize_images,
    show_first_batch,
    show_images,
    show_time_series,
)

_STYLE_DIR = files("smon_toolbox.vis")


def use_style(style="academic"):
    """Activate a bundled matplotlib style.

    Args:
        style: Name of the style file (without .mplstyle extension).
               Default is "academic" for publication-quality figures.
    """
    style_path = _STYLE_DIR / f"{style}.mplstyle"
    plt.style.use(str(style_path))
