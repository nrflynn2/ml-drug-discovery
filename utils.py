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
    save_dir = Path(f"data/{chapter}/artifacts")
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
    # Create the full file path
    file_dir = Path(f"data/{chapter}/artifacts")
    
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
    save_dir = Path(f"data/{chapter}/artifacts")
    
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