#!/usr/bin/env python3
"""
Test script for the config.py module.
Tests all configuration functions and verifies environment variable handling.
"""

import sys
import os
from pathlib import Path

# Add scripts/config to path to import config module
sys.path.insert(0, str(Path(__file__).parent.parent / 'config'))

try:
    import config
    print("✓ Successfully imported config module")
except ImportError as e:
    print(f"✗ Failed to import config module: {e}")
    sys.exit(1)

def test_config_functions():
    """Test all configuration functions"""
    print("\n" + "="*60)
    print("TESTING CONFIG FUNCTIONS")
    print("="*60)

    # Test basic functions
    functions_to_test = [
        ('get_git_root', config.get_git_root),
        ('get_local_root', config.get_local_root),
        ('get_data_dir', config.get_data_dir),
        ('get_models_dir', config.get_models_dir),
        ('get_temp_dir', config.get_temp_dir),
        ('get_default_model_name', config.get_default_model_name),
        ('get_default_model_path', config.get_default_model_path),
        ('get_raw_mseed_events_dir', lambda: config.get_raw_mseed_events_dir('2024')),
        ('get_processed_mseed_events_dir', lambda: config.get_processed_mseed_events_dir('test_1000')),
        ('get_results_gpd_keras_dir', config.get_results_gpd_keras_dir),
        ('get_analysis_plots_dir', config.get_analysis_plots_dir),
    ]

    for func_name, func in functions_to_test:
        try:
            result = func()
            print(f"✓ {func_name:25} -> {result}")
        except Exception as e:
            print(f"✗ {func_name:25} -> ERROR: {e}")

def test_with_subdirectories():
    """Test get_data_dir with subdirectories"""
    print("\n" + "-"*60)
    print("TESTING SUBDIRECTORIES")
    print("-"*60)

    subdirs = ['raw', 'preprocessed', 'results', 'models', 'temp']
    for subdir in subdirs:
        try:
            result = config.get_data_dir(subdir)
            print(f"✓ get_data_dir('{subdir}'):     {result}")
        except Exception as e:
            print(f"✗ get_data_dir('{subdir}'):     ERROR: {e}")

def test_environment_variables():
    """Test behavior with different environment variable combinations"""
    print("\n" + "-"*60)
    print("TESTING ENVIRONMENT VARIABLES")
    print("-"*60)

    # Save original environment
    original_env = {}
    env_vars = ['GPD_GIT_ROOT', 'GPD_LOCAL_ROOT', 'GPD_DATA_DIR', 'GPD_MODELS_DIR', 'GPD_TEMP_DIR', 'GPD_DEFAULT_MODEL']

    for var in env_vars:
        original_env[var] = os.environ.get(var)

    try:
        # Test with custom environment variables
        test_env = {
            'GPD_GIT_ROOT': '/custom/git/root',
            'GPD_LOCAL_ROOT': '/custom/local/root',
            'GPD_DATA_DIR': '/custom/data',
            'GPD_MODELS_DIR': '/custom/models',
            'GPD_TEMP_DIR': '/custom/temp',
            'GPD_DEFAULT_MODEL': 'custom_model.keras'
        }

        print("Setting custom environment variables:")
        for var, value in test_env.items():
            os.environ[var] = value
            print(f"  {var} = {value}")

        print("\nResults with custom environment:")
        print(f"  get_git_root():         {config.get_git_root()}")
        print(f"  get_local_root():       {config.get_local_root()}")
        print(f"  get_data_dir():         {config.get_data_dir()}")
        print(f"  get_models_dir():       {config.get_models_dir()}")
        print(f"  get_temp_dir():         {config.get_temp_dir()}")
        print(f"  get_default_model():    {config.get_default_model_name()}")

    finally:
        # Restore original environment
        for var in env_vars:
            if original_env[var] is not None:
                os.environ[var] = original_env[var]
            elif var in os.environ:
                del os.environ[var]

def test_directory_creation():
    """Test ensure_directories function (dry run)"""
    print("\n" + "-"*60)
    print("TESTING DIRECTORY CREATION (DRY RUN)")
    print("-"*60)

    try:
        # Show what directories would be created
        directories = [
            config.get_git_root(),
            config.get_local_root(),
            config.get_data_dir(),
            config.get_models_dir(),
            config.get_temp_dir(),
            config.get_data_dir('raw'),
            config.get_data_dir('preprocessed'),
            config.get_data_dir('results'),
        ]

        print("Directories that would be ensured:")
        for directory in directories:
            exists = "EXISTS" if directory.exists() else "MISSING"
            print(f"  {exists:7} -> {directory}")

        # Note: Not actually calling ensure_directories() to avoid creating dirs
        print("\nNote: ensure_directories() not called to avoid creating directories")

    except Exception as e:
        print(f"✗ Directory creation test failed: {e}")

def test_utility_functions():
    """Test utility functions"""
    print("\n" + "-"*60)
    print("TESTING UTILITY FUNCTIONS")
    print("-"*60)

    try:
        # Test get_relative_to_data
        data_dir = config.get_data_dir()
        test_path = data_dir / 'test' / 'subdir' / 'file.txt'
        relative_path = config.get_relative_to_data(test_path)
        print(f"✓ get_relative_to_data({test_path}) -> {relative_path}")

        # Test with path outside data dir
        outside_path = Path('/tmp/outside.txt')
        relative_outside = config.get_relative_to_data(outside_path)
        print(f"✓ get_relative_to_data({outside_path}) -> {relative_outside}")

    except Exception as e:
        print(f"✗ Utility functions test failed: {e}")

def main():
    """Run all tests"""
    print("GPD CONFIG.PY TEST SCRIPT")
    print("=" * 60)

    # Show current environment
    print("Current environment variables:")
    env_vars = ['GPD_GIT_ROOT', 'GPD_LOCAL_ROOT', 'GPD_DATA_DIR', 'GPD_MODELS_DIR', 'GPD_TEMP_DIR', 'GPD_DEFAULT_MODEL']
    for var in env_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"  {var:20} = {value}")

    # Run tests
    test_config_functions()
    test_with_subdirectories()
    test_environment_variables()
    test_directory_creation()
    test_utility_functions()

    # Show final configuration summary
    print("\n" + "="*60)
    print("FINAL CONFIGURATION SUMMARY")
    print("="*60)
    config.print_config()

    print("\n✓ All tests completed!")

if __name__ == '__main__':
    main()