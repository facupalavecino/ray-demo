import os

from pathlib import Path

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parents[1].resolve()
"""Project's root directory"""

DATA_DIR = ROOT_DIR / "data"
"""Project's data directory"""
