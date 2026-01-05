from importlib.metadata import version

__version__ = version("smon-toolbox")

# Optional: Expose submodules for easier access
# This allows: import smon_toolbox as sm; sm.linear.cholesky(...)
from . import linear, networks, odes, optimization, utils
