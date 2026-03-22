"""
seer_peph
=========
Bayesian spatial joint treatment–survival model for SEER-Medicare cancer data.

Piecewise Exponential Proportional Hazards (PEPH) implementation using NumPyro.

Sub-packages
------------
data      — data loading, encoding, and long-format expansion
"""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("SEER-PEPH")
except PackageNotFoundError:
    __version__ = "dev"