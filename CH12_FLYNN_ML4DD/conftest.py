"""Pytest bootstrap for the CH12 protein-transformer package.

The package uses top-level ``import`` statements like ``from src.utils import ...``,
which requires this directory (the one containing ``src/``) to be on ``sys.path``.
Placing this conftest at the package root makes pytest prepend it automatically,
so ``pytest`` can be run from anywhere (``pytest CH12_FLYNN_ML4DD``).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
