"""
Configuration module for GPD data directories and paths.
Handles environment variable-based configuration for data storage.
"""

import os
from pathlib import Path
from typing import Optional

# Load environment variables from .env file if it exists
def _load_env_file():
    """Load environment variables from .env file if it exists"""
    # Find .env file starting from script location up to git root
    current_dir = Path(__file__).parent
    env_vars = {}

    for _ in range(5):  # Check up to 5 levels up
        env_file = current_dir / '.env'
        if env_file.exists():
            # First pass: load all variables
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key, value = key.strip(), value.strip()
                        env_vars[key] = value

            # Second pass: expand variables and set environment
            for key, value in env_vars.items():
                # Expand ${VAR} syntax
                expanded_value = value
                import re
                for match in re.findall(r'\$\{([^}]+)\}', value):
                    if match in env_vars:
                        expanded_value = expanded_value.replace(f'${{{match}}}', env_vars[match])
                    elif match in os.environ:
                        expanded_value = expanded_value.replace(f'${{{match}}}', os.environ[match])

                # Only set if not already in environment
                if key not in os.environ:
                    os.environ[key] = expanded_value
            break
        current_dir = current_dir.parent
        if current_dir == current_dir.parent:  # Reached root
            break

# Load .env file on import
_load_env_file()

def get_git_root() -> Path:
    """Get the git repository root directory"""
    return Path(os.getenv('GPD_GIT_ROOT', '.'))

def get_local_root() -> Path:
    """Get the local project root directory"""
    return Path(os.getenv('GPD_LOCAL_ROOT', './'))

def get_data_dir(subdir: Optional[str] = None) -> Path:
    """
    Get the main data directory or a subdirectory within it.

    Args:
        subdir: Optional subdirectory name (e.g., 'raw', 'preprocessed', 'results')

    Returns:
        Path object for the requested directory
    """
    base_dir = Path(os.getenv('GPD_DATA_DIR', get_local_root() / 'data'))

    if subdir:
        return base_dir / subdir
    return base_dir


def get_models_dir() -> Path:
    """Get directory for trained models"""
    return Path(os.getenv('GPD_MODELS_DIR', get_git_root() / 'models'))

def get_temp_dir() -> Path:
    """Get directory for temporary processing files"""
    return Path(os.getenv('GPD_TEMP_DIR', get_local_root() / 'temp'))

def get_default_model_path() -> Path:
    """Get path to the default GPD model"""
    model_name = os.getenv('GPD_DEFAULT_MODEL', 'gpd_v2.keras')
    return get_models_dir() / model_name

def get_default_model_name() -> str:
    """Get the name of the default GPD model"""
    return os.getenv('GPD_DEFAULT_MODEL', 'gpd_v2.keras')

def ensure_directories():
    """
    Create all configured directories if they don't exist.
    Call this function during initialization to ensure directory structure.
    """
    directories = [
        get_git_root(),
        get_local_root(),
        get_data_dir(),
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
    print("GPD Directory Configuration:")
    print(f"  Git root:        {get_git_root()}")
    print(f"  Local root:      {get_local_root()}")
    print(f"  Main data dir:   {get_data_dir()}")
    print(f"  Models dir:      {get_models_dir()}")
    print(f"  Temp dir:        {get_temp_dir()}")
    print(f"  Default model:   {get_default_model_name()}")
    print(f"  Model path:      {get_default_model_path()}")