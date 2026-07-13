import os
import pickle
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from pathlib import Path

def save_molecular_dataframe(df, filename, chapter="ch01", compress=True):
    """
    Save a pandas DataFrame containing molecular data to a pickle file.
    
    This function handles dataframes that contain mixed data types including
    RDKit Mol objects, which cannot be saved with standard methods like
    df.to_csv() or df.to_parquet().
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to save, which may contain RDKit Mol objects
    filename : str
        Name of the file (without path)
    chapter : str
        Chapter identifier for directory organization
    compress : bool
        Whether to use compression (recommended for large dataframes)
    
    Returns:
    --------
    str
        Path to the saved file
    """
    # Create the artifacts directory if it doesn't exist
    save_dir = Path(f"artifacts/{chapter}/")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Add .pkl or .pkl.gz extension if not present
    if not filename.endswith('.pkl') and not filename.endswith('.pkl.gz'):
        filename = f"{filename}.pkl"
    
    # Add compression extension if requested
    if compress and not filename.endswith('.gz'):
        filename = f"{filename}.gz"
    
    # Full path for saving
    save_path = save_dir / filename
    
    # Save the dataframe using pickle with optional compression
    protocol = pickle.HIGHEST_PROTOCOL  # Use the most efficient protocol
    
    print(f"Saving dataframe with {len(df)} rows to {save_path}...")
    with open(save_path, 'wb') as f:
        pickle.dump(df, f, protocol=protocol)
    
    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Successfully saved dataframe ({file_size_mb:.1f} MB)")
    
    return str(save_path)

def load_molecular_dataframe(filename, chapter="ch01"):
    """
    Load a pandas DataFrame containing molecular data from a pickle file.
    
    Parameters:
    -----------
    filename : str
        Name of the file to load (without path)
    chapter : str
        Chapter identifier for directory organization
    
    Returns:
    --------
    pandas.DataFrame
        The loaded DataFrame
    """
    # Create the full file path, anchored to the repo root (the directory
    # containing utils.py) so that it works regardless of the caller's CWD.
    # The notebook kernel's CWD can land in arbitrary subdirectories (e.g.
    # data/ch02/) when cells change directories mid-run, which previously
    # caused a FileNotFoundError at runtime.
    repo_root = Path(__file__).resolve().parent
    file_dir = repo_root / "artifacts" / chapter
    
    # Handle different possible file extensions
    if not (filename.endswith('.pkl') or filename.endswith('.pkl.gz')):
        # Try both compressed and uncompressed versions
        if (file_dir / f"{filename}.pkl.gz").exists():
            filename = f"{filename}.pkl.gz"
        else:
            filename = f"{filename}.pkl"
    
    file_path = file_dir / filename
    
    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    print(f"Loading molecular dataframe from {file_path}...")
    start_time = pd.Timestamp.now()
    
    # Load the dataframe
    with open(file_path, 'rb') as f:
        df = pickle.load(f)
    
    # Calculate loading time
    load_time = (pd.Timestamp.now() - start_time).total_seconds()
    
    print(f"Successfully loaded dataframe with {len(df)} rows and {len(df.columns)} columns")
    print(f"Loading time: {load_time:.2f} seconds")
    
    return df

def list_saved_dataframes(chapter="ch01"):
    """
    List all saved dataframes in the artifacts directory.
    
    Parameters:
    -----------
    chapter : str
        Chapter identifier for directory organization
    
    Returns:
    --------
    list
        List of available dataframe filenames
    """
    save_dir = Path(f"artifacts/{chapter}/")
    
    if not save_dir.exists():
        print(f"No artifacts directory found for chapter {chapter}")
        return []
    
    # Get all pickle files
    saved_files = list(save_dir.glob("*.pkl*"))
    
    if not saved_files:
        print(f"No saved dataframes found in {save_dir}")
        return []
    
    # Print information about available files
    print(f"Available saved dataframes in {save_dir}:")
    file_info = []
    
    for file_path in saved_files:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        modified_time = pd.Timestamp.fromtimestamp(os.path.getmtime(file_path))
        
        file_info.append({
            'filename': file_path.name,
            'size_mb': file_size_mb,
            'modified': modified_time
        })
        
    # Sort by modification time (newest first)
    file_info.sort(key=lambda x: x['modified'], reverse=True)

    # Display the information
    for info in file_info:
        print(f"  {info['filename']} ({info['size_mb']:.1f} MB, modified: {info['modified']})")

    return [info['filename'] for info in file_info]


def set_seed(seed=42, deterministic=False):
    """
    Seed Python, NumPy, and (if installed) PyTorch + CUDA for reproducibility.

    Call once near the top of a notebook, e.g. ``SEED = set_seed(42)``. Seeding
    every RNG the notebooks touch -- plus ``PYTHONHASHSEED`` -- is what makes the
    printed outputs reproducible across runs and machines.

    Parameters
    ----------
    seed : int
        The seed applied to all random number generators.
    deterministic : bool
        If True, also force deterministic cuDNN / PyTorch kernels. This can slow
        training and some ops have no deterministic implementation, so it is
        opt-in (``warn_only=True`` avoids hard failures on those ops).

    Returns
    --------
    int
        The seed, so you can capture it in one line: ``SEED = set_seed(42)``.
    """
    import random

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            import torch.backends.cudnn as cudnn
            cudnn.deterministic = True
            cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:
        pass

    return seed


def preview_df(df, name="df", n=3):
    """
    Print a compact before/after snapshot of a DataFrame and return its head.

    Standardizes the "show shape + columns + head after each major transform"
    pattern so readers can see exactly how the data changes from step to step
    (raw -> parsed -> descriptors -> filters -> fingerprints -> hits).

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to preview.
    name : str
        A label for the printout (e.g. "raw", "filtered", "with descriptors").
    n : int
        Number of head rows to return for display.

    Returns
    --------
    pandas.DataFrame
        ``df.head(n)`` so the call renders a preview as the cell's output.
    """
    print(f"{name}: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"  columns: {list(df.columns)}")
    return df.head(n)


def check_env(packages=None):
    """
    Print the Python version and key package versions for reproducibility.

    Several listings are version-sensitive (RDKit descriptor counts, scikit-learn
    calibration APIs, PyTorch schedulers). Printing versions up front makes a
    notebook's results self-documenting and much easier to debug across the
    local, WSL2, and Colab environments the book supports.

    Parameters
    ----------
    packages : list of str, optional
        Import names to report. Defaults to the packages the book relies on.
    """
    import importlib
    import platform

    if packages is None:
        packages = [
            "numpy", "pandas", "scipy", "sklearn", "rdkit",
            "torch", "torch_geometric", "xgboost", "matplotlib",
        ]

    print(f"Python {platform.python_version()} ({platform.system()})")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA available: no (CPU only)")
    except ImportError:
        pass

    for name in packages:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {name:16s} {version}")
        except ImportError:
            print(f"  {name:16s} (not installed)")


def get_device():
    """
    Return the best available PyTorch device, hardware-agnostic (CUDA / Apple MPS /
    other accelerator, else CPU).

    Prefers ``torch.accelerator`` (PyTorch >= 2.6), which unifies detection across
    CUDA, Apple MPS, Intel XPU, and others -- the pattern contributed by
    thomas-to-bcheme (GitHub PRs #24-28). Falls back to explicit CUDA/MPS checks on
    older PyTorch or if the accelerator query fails. Use alongside ``set_seed()`` in
    the standardized setup cell so every deep-learning chapter runs on whatever
    hardware the reader has (GPU, Mac, or CPU).

    Returns
    --------
    torch.device
        The selected device.
    """
    import torch

    # torch.accelerator (>= 2.6) unifies CUDA / MPS / XPU / ... detection.
    if hasattr(torch, "accelerator"):
        try:
            if torch.accelerator.is_available():
                return torch.device(torch.accelerator.current_accelerator())
        except Exception:
            pass

    # Fallback for older PyTorch (or if the accelerator query is unavailable).
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")