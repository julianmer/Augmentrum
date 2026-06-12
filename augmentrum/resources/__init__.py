"""
augmentrum.resources — bundled reference data shipped with the package.

Use get_scans_dir() to get a pathlib.Path to the scans folder regardless
of how / where augmentrum was installed.
"""

from importlib.resources import files
from pathlib import Path


def get_scans_dir() -> Path:
    """Return the path to the bundled augmentrum/resources/scans/ directory."""
    return Path(str(files("augmentrum.resources.scans")))
