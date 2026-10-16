"""crypto pairs trading research, not a brokerage interface."""
from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("crypto-stat-arb-lab")
except PackageNotFoundError:  # running from a source checkout without an install
    __version__ = "0.0.0+unknown"
