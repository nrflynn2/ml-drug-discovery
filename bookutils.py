"""Shared utilities for *Build AI Drug Discovery Pipelines*.

One import surface for every chapter notebook. Consolidating this here kills the
copy-paste drift the 3P review flagged: a single canonical seeder, one house
plotting style, one device helper, one figure-saving policy, and one Colab
environment bootstrap.

Design notes
------------
* Heavy, tier-specific dependencies (``torch``, ``rdkit``, ``matplotlib``) are
  imported lazily *inside* the functions that need them, so ``import bookutils``
  works even in the ``core`` install tier (Chapters 1-4), which has no torch.
* All on-disk paths are anchored to the repository root (the directory holding
  this file) so they are independent of the notebook kernel's working directory.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np

# Re-export the anchored dataframe helpers so a chapter imports a single module.
from utils import (  # noqa: F401
    load_molecular_dataframe,
    save_molecular_dataframe,
    list_saved_dataframes,
)

__all__ = [
    "REPO_ROOT",
    "SEED",
    "SCREEN_DPI",
    "SAVE_DPI",
    "set_seed",
    "setup_style",
    "get_device",
    "setup_rdkit_drawing",
    "save_figure",
    "in_colab",
    "setup_environment",
    "load_molecular_dataframe",
    "save_molecular_dataframe",
    "list_saved_dataframes",
]

REPO_ROOT = Path(__file__).resolve().parent

# Book-wide reproducibility seed. Every chapter seeds with this value.
SEED = 42

# Single DPI policy: crisp-but-light on screen, print-quality when saved.
SCREEN_DPI = 100
SAVE_DPI = 300

# House categorical palette (colour-blind-safe, used across every chapter).
PALETTE = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # grey
]


def set_seed(seed: int = SEED) -> None:
    """Seed every RNG that affects reproducibility.

    Seeds Python's ``random``, NumPy, and — when available — PyTorch (CPU and all
    CUDA devices), plus ``PYTHONHASHSEED`` and cuDNN's deterministic flags. This
    is the one canonical seeder for the whole book; call it once, right after the
    imports, in every chapter.

    Args:
        seed: The seed value. Defaults to the book-wide :data:`SEED` (42).
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_style() -> None:
    """Apply the single house plotting theme (palette, figure size, screen DPI).

    Uses seaborn if present, otherwise falls back to Matplotlib's rcParams so the
    theme still applies in the ``core`` tier.
    """
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns

        sns.set_theme(style="whitegrid", palette=PALETTE)
    except ImportError:
        pass

    plt.rcParams.update(
        {
            "figure.figsize": (8, 5),
            "figure.dpi": SCREEN_DPI,
            "savefig.dpi": SAVE_DPI,
            "savefig.bbox": "tight",
            "axes.prop_cycle": plt.cycler(color=PALETTE),
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "font.size": 11,
        }
    )


def get_device(verbose: bool = True):
    """Return the best available torch device and (optionally) print a banner.

    Returns:
        torch.device: ``cuda`` if a GPU is visible, else ``cpu``.
    """
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        if verbose:
            print("Using CPU (no CUDA device found).")
    return device


def setup_rdkit_drawing() -> None:
    """Enable inline RDKit molecule drawing with consistent defaults."""
    from rdkit.Chem.Draw import IPythonConsole

    IPythonConsole.ipython_useSVG = True
    IPythonConsole.molSize = (400, 300)


def save_figure(fig, name: str, chapter: str, formats=("png", "pdf")) -> list[str]:
    """Save a Matplotlib figure to ``figures/<chapter>/`` in PNG and PDF.

    Anchored to the repo root, so it works regardless of the kernel CWD. Only
    ~half the chapters saved figures before; this makes it one call everywhere.

    Args:
        fig: The Matplotlib ``Figure`` to save.
        name: Base filename (without extension).
        chapter: Chapter directory key, e.g. ``"ch01"``.
        formats: Iterable of extensions to write. Defaults to PNG + PDF.

    Returns:
        The list of written file paths (as strings).
    """
    out_dir = REPO_ROOT / "figures" / chapter
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in formats:
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, dpi=SAVE_DPI, bbox_inches="tight")
        written.append(str(path))
    return written


def in_colab() -> bool:
    """True when running inside Google Colab."""
    return "google.colab" in sys.modules


def setup_environment(chapter: str, tier: str = "core") -> Path:
    """Standardize per-chapter directory layout (and Colab data staging).

    Creates ``data/<chapter>``, ``artifacts/<chapter>`` and ``figures/<chapter>``
    under the repo root and returns the chapter's data directory. On Colab the
    repo root resolves to the cloned checkout. Package installation itself stays
    in the notebook's tier-specific pip/conda cell — this helper only owns the
    directory + path bootstrap so those setup cells stop drifting.

    Args:
        chapter: Chapter directory key, e.g. ``"ch09"``.
        tier: Install tier name (``core``/``advanced``/``full``/``conda``),
            recorded for the printed banner; installation is done by the
            notebook cell, not here.

    Returns:
        Path to ``data/<chapter>``.
    """
    for sub in ("data", "artifacts", "figures"):
        (REPO_ROOT / sub / chapter).mkdir(parents=True, exist_ok=True)

    where = "Google Colab" if in_colab() else "local environment"
    print(f"Environment ready for {chapter} ({tier} tier) in {where}.")
    return REPO_ROOT / "data" / chapter
