"""
Configuration module for GPD data directories and paths.
Handles environment variable-based configuration for data storage.
"""

import os
from pathlib import Path
from typing import Optional

def get_data_dir(subdir: Optional[str] = None) -> Path:
    """
    Get the main data directory or a subdirectory within it.

    Args:
        subdir: Optional subdirectory name (e.g., 'raw', 'preprocessed', 'results')

    Returns:
        Path object for the requested directory
    """
    base_dir = Path(os.getenv('GPD_DATA_DIR', './data'))

    if subdir:
        return base_dir / subdir
    return base_dir

def get_raw_data_dir() -> Path:
    """Get directory for raw seismic data files (.mseed)"""
    return Path(os.getenv('GPD_RAW_DIR', get_data_dir('raw')))

def get_preprocessed_data_dir() -> Path:
    """Get directory for preprocessed seismic data"""
    return Path(os.getenv('GPD_PREPROCESSED_DIR', get_data_dir('preprocessed')))

def get_results_dir() -> Path:
    """Get directory for inference results and outputs"""
    return Path(os.getenv('GPD_RESULTS_DIR', get_data_dir('results')))

def get_models_dir() -> Path:
    """Get directory for trained models"""
    return Path(os.getenv('GPD_MODELS_DIR', get_data_dir('models')))

def get_temp_dir() -> Path:
    """Get directory for temporary processing files"""
    return Path(os.getenv('GPD_TEMP_DIR', get_data_dir('temp')))

def ensure_directories():
    """
    Create all configured directories if they don't exist.
    Call this function during initialization to ensure directory structure.
    """
    directories = [
        get_data_dir(),
        get_raw_data_dir(),
        get_preprocessed_data_dir(),
        get_results_dir(),
        get_models_dir(),
        get_temp_dir()
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

def get_relative_to_data(file_path: Path) -> Path:
    """
    Convert an absolute path to relative path from data directory.
    Useful for logging and display purposes.
    """
    try:
        return file_path.relative_to(get_data_dir())
    except ValueError:
        return file_path

# Configuration summary for debugging
def print_config():
    """Print current configuration for debugging purposes"""
    print("GPD Data Directory Configuration:")
    print(f"  Main data dir: {get_data_dir()}")
    print(f"  Raw data dir:  {get_raw_data_dir()}")
    print(f"  Preprocessed:  {get_preprocessed_data_dir()}")
    print(f"  Results dir:   {get_results_dir()}")
    print(f"  Models dir:    {get_models_dir()}")
    print(f"  Temp dir:      {get_temp_dir()}")