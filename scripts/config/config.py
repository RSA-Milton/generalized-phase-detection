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

            # Multiple passes to expand nested variables
            import re
            max_iterations = 10
            for iteration in range(max_iterations):
                changes_made = False

                for key, value in env_vars.items():
                    # Skip if already fully expanded (no ${} patterns)
                    if '${' not in value:
                        continue

                    # Expand ${VAR} syntax
                    expanded_value = value
                    for match in re.findall(r'\$\{([^}]+)\}', value):
                        replacement = None

                        # First try from env_vars (current file)
                        if match in env_vars and '${' not in env_vars[match]:
                            replacement = env_vars[match]
                        # Then try from environment
                        elif match in os.environ:
                            replacement = os.environ[match]

                        if replacement:
                            expanded_value = expanded_value.replace(f'${{{match}}}', replacement)
                            changes_made = True

                    # Update the value
                    env_vars[key] = expanded_value

                # If no changes were made, we're done
                if not changes_made:
                    break

            # Set all expanded variables in environment
            for key, value in env_vars.items():
                if key not in os.environ:
                    os.environ[key] = value
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

# Nuevas funciones para estructura de datos mejorada
def get_raw_data_dir(subdir: Optional[str] = None) -> Path:
    """Get raw data directory"""
    base_dir = Path(os.getenv('GPD_RAW_DIR', get_data_dir('raw')))
    if subdir:
        return base_dir / subdir
    return base_dir

def get_processed_data_dir(subdir: Optional[str] = None) -> Path:
    """Get processed data directory"""
    base_dir = Path(os.getenv('GPD_PROCESSED_DIR', get_data_dir('processed')))
    if subdir:
        return base_dir / subdir
    return base_dir

def get_results_data_dir(subdir: Optional[str] = None) -> Path:
    """Get results data directory"""
    base_dir = Path(os.getenv('GPD_RESULTS_DIR', get_data_dir('results')))
    if subdir:
        return base_dir / subdir
    return base_dir

def get_analysis_data_dir(subdir: Optional[str] = None) -> Path:
    """Get analysis data directory"""
    base_dir = Path(os.getenv('GPD_ANALYSIS_DIR', get_data_dir('analysis')))
    if subdir:
        return base_dir / subdir
    return base_dir

# Funciones específicas para subdirectorios detallados
def get_raw_mseed_dir(subdir: Optional[str] = None) -> Path:
    """Get raw MSEED directory"""
    base_dir = Path(os.getenv('GPD_RAW_MSEED_DIR', get_raw_data_dir('mseed')))
    if subdir:
        return base_dir / subdir
    return base_dir

def get_raw_mseed_continuos_dir(year: Optional[str] = None) -> Path:
    """Get raw MSEED continuous data directory"""
    base_dir = Path(os.getenv('GPD_RAW_MSEED_CONTINUOS_DIR', get_raw_mseed_dir('continuos')))
    if year:
        return base_dir / year
    return base_dir

def get_raw_mseed_events_dir(year: Optional[str] = None) -> Path:
    """Get raw MSEED events directory"""
    base_dir = Path(os.getenv('GPD_RAW_MSEED_EVENTS_DIR', get_raw_mseed_dir('events')))
    if year:
        return base_dir / year
    return base_dir

def get_raw_mseed_waveforms_dir() -> Path:
    """Get raw MSEED waveforms directory"""
    return Path(os.getenv('GPD_RAW_MSEED_WAVEFORMS_DIR', get_raw_mseed_dir('waveforms')))

def get_raw_mseed_legacy_dir() -> Path:
    """Get raw MSEED legacy directory"""
    return Path(os.getenv('GPD_RAW_MSEED_LEGACY_DIR', get_raw_mseed_dir('legacy')))

def get_raw_datasets_dir() -> Path:
    """Get raw datasets directory"""
    return Path(os.getenv('GPD_RAW_DATASETS_DIR', get_raw_data_dir('datasets')))

def get_processed_mseed_dir(subdir: Optional[str] = None) -> Path:
    """Get processed MSEED directory"""
    base_dir = Path(os.getenv('GPD_PROCESSED_MSEED_DIR', get_processed_data_dir('mseed')))
    if subdir:
        return base_dir / subdir
    return base_dir

def get_processed_mseed_continuos_dir(year: Optional[str] = None) -> Path:
    """Get processed MSEED continuous data directory"""
    base_dir = Path(os.getenv('GPD_PROCESSED_MSEED_CONTINUOS_DIR', get_processed_mseed_dir('continuos')))
    if year:
        return base_dir / year
    return base_dir

def get_processed_mseed_events_dir(dataset: Optional[str] = None) -> Path:
    """Get processed MSEED events directory"""
    base_dir = Path(os.getenv('GPD_PROCESSED_MSEED_EVENTS_DIR', get_processed_mseed_dir('events')))
    if dataset:
        return base_dir / dataset
    return base_dir

def get_processed_datasets_dir() -> Path:
    """Get processed datasets directory"""
    return Path(os.getenv('GPD_PROCESSED_DATASETS_DIR', get_processed_data_dir('datasets')))

def get_results_gpd_dir(method: Optional[str] = None) -> Path:
    """Get GPD results directory"""
    base_dir = Path(os.getenv('GPD_RESULTS_GPD_DIR', get_results_data_dir('gpd')))
    if method:
        return base_dir / method
    return base_dir

def get_results_gpd_keras_dir() -> Path:
    """Get GPD Keras results directory"""
    return Path(os.getenv('GPD_RESULTS_GPD_KERAS_DIR', get_results_gpd_dir('keras')))

def get_results_gpd_legacy_dir() -> Path:
    """Get GPD legacy results directory"""
    return Path(os.getenv('GPD_RESULTS_GPD_LEGACY_DIR', get_results_gpd_dir('legacy')))

def get_results_gpd_tflite_dir() -> Path:
    """Get GPD TFLite results directory"""
    return Path(os.getenv('GPD_RESULTS_GPD_TFLITE_DIR', get_results_gpd_dir('tflite')))

def get_results_stalta_dir() -> Path:
    """Get STA/LTA results directory"""
    return Path(os.getenv('GPD_RESULTS_STALTA_DIR', get_results_data_dir('stalta')))

def get_results_comparisons_dir() -> Path:
    """Get comparisons results directory"""
    return Path(os.getenv('GPD_RESULTS_COMPARISONS_DIR', get_results_data_dir('comparisons')))

def get_analysis_performance_dir() -> Path:
    """Get performance analysis directory"""
    return Path(os.getenv('GPD_ANALYSIS_PERFORMANCE_DIR', get_analysis_data_dir('performance')))

def get_analysis_statistics_dir() -> Path:
    """Get statistics analysis directory"""
    return Path(os.getenv('GPD_ANALYSIS_STATISTICS_DIR', get_analysis_data_dir('statistics')))

def get_analysis_plots_dir() -> Path:
    """Get plots analysis directory"""
    return Path(os.getenv('GPD_ANALYSIS_PLOTS_DIR', get_analysis_data_dir('plots')))

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
        get_temp_dir(),
        get_raw_data_dir(),
        get_processed_data_dir(),
        get_results_data_dir(),
        get_analysis_data_dir()
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
    print(f"  Raw data:        {get_raw_data_dir()}")
    print(f"  Processed data:  {get_processed_data_dir()}")
    print(f"  Results data:    {get_results_data_dir()}")
    print(f"  Analysis data:   {get_analysis_data_dir()}")
    print(f"  Default model:   {get_default_model_name()}")
    print(f"  Model path:      {get_default_model_path()}")