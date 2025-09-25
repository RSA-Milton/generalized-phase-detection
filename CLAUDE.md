# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research repository for Generalized Phase Detection (GPD), a deep learning framework for seismic phase detection developed by Ross et al. (2018). The codebase contains Python scripts for model conversion, inference, validation, and analysis of seismic data.

## Environment Setup

### Conda Environments

The project uses multiple conda environments for different purposes:

- **gpd_py39** (primary): For modern inference with TensorFlow/Keras
- **gpd_py36**: For legacy inference with older TensorFlow versions
- **stalta_py39**: For STA/LTA detection algorithms

Environment files are stored in `env/` directory as `.lock` files. To recreate environments:

```bash
micromamba activate gpd_py39        # For modern inference
micromamba activate gpd_py36        # For legacy inference
micromamba activate stalta_py39     # For STA/LTA detection
```

### Data Directory Configuration

The project uses environment variables for data directory configuration. Copy `.env.example` to `.env` and modify paths:

```bash
# Example configuration
GPD_DATA_DIR=/home/user/projects/gpd/data
GPD_MODELS_DIR=/home/user/git/generalized-phase-detection/models
GPD_TEMP_DIR=/home/user/projects/gpd/temp
```

Available environment variables:
- `GPD_DATA_DIR`: Main data directory (default: `./data`)
- `GPD_RAW_DIR`: Raw seismic data files directory
- `GPD_PREPROCESSED_DIR`: Preprocessed data directory
- `GPD_RESULTS_DIR`: Inference results directory
- `GPD_MODELS_DIR`: Trained models directory
- `GPD_TEMP_DIR`: Temporary processing files

## Important Notes for Claude Code

⚠️ **Environment Restrictions**: Most scripts in the `scripts/` directory require specific conda environments and cannot be executed directly by Claude Code. Only scripts in `scripts/utils/` can be run without environment issues.

For testing and validation of scripts outside `scripts/utils/`, use the following approach:
- Review code structure and imports
- Test configuration imports separately: `python3 -c "import sys; sys.path.insert(0, 'scripts/config'); import config; config.print_config()"`
- Validate argument parsing: `python3 script.py --help` (may fail due to missing dependencies)
- Focus on code analysis rather than execution

## Common Commands

### Data Preprocessing
**Note: These commands require conda environment `gpd_py39`**
```bash
# Preprocess individual mseed file for GPD
micromamba activate gpd_py39
python scripts/mseed/preprocess_mseed_for_gpd.py -I input.mseed -O output.mseed --input-freq 64 -V

# Batch processing of directory
python scripts/mseed/preprocess_mseed_for_gpd.py -I input_dir/ -O output_dir/ --input-freq 64 --batch -V

# Generate dataset with SNR information
python scripts/mseed/mseed_for_gpd.py -I dataset.csv -O output_dir --input-freq 64 -V
```

### Inference
**Note: These commands require specific conda environments**
```bash
# Keras inference (requires gpd_py39)
micromamba activate gpd_py39

# Individual file inference
python scripts/inference/chunked/gpd_keras_inference_chunked.py -I input.mseed -O output.out -V

# Batch inference with custom thresholds (uses .env configuration)
python scripts/inference/gpd_keras_inference_events.py --min-proba-p 0.55 --min-proba-s 0.85 -V

# Use different model (specify name only, path resolved automatically)
python scripts/inference/gpd_keras_inference_events.py --model-path gpd_v1.keras -V

# Legacy inference (requires gpd_py36)
micromamba activate gpd_py36
python scripts/inference/chunked/gpd_legacy_inference_chunked.py -I input.mseed -O output.out -V

# STA/LTA detection (requires stalta_py39)
micromamba activate stalta_py39
python scripts/inference/stalta_inference_events.py -I input_dir -O output.csv -V
```

### Analysis and Comparison
**Note: These commands require conda environment `gpd_py39`**
```bash
micromamba activate gpd_py39

# Compare GPD results with analyst picks
python scripts/debug/compare_analyst-automated_events.py --analyst analyst_picks.csv --automated gpd_results.csv --tolerance 0.5 1.0 2.0 5.0 -V

# Interactive GUI tools for data inspection
python scripts/gui/mseed_event_inspector.py  # Interactive event inspection
python scripts/gui/mseed_event_extractor.py  # Event extraction GUI
```

### Testing and Configuration (Claude Code Compatible)
**These scripts can be executed by Claude Code without environment issues:**
```bash
# Test configuration system
python3 scripts/utils/test_config.py

# Validate .env file loading and variable expansion
python3 -c "import sys; sys.path.insert(0, 'scripts/config'); import config; config.print_config()"
```

## Architecture

### Core Directories

- **`scripts/`**: Main processing scripts organized by function
  - `conversion_modelos/`: Model format conversion tools (HDF5, SavedModel, TFLite)
  - `inference/`: Batch inference and evaluation scripts
  - `mseed/`: MSEED data processing and preprocessing
  - `utils/`: Analysis and comparison utilities
  - `gui/`: Interactive data inspection tools

- **`scripts/inference/`**: Inference and detection scripts
  - `chunked/`: Individual file inference implementations (Keras, legacy, TFLite)
  - Batch processing and evaluation scripts
  - STA/LTA detector implementation

- **`legacy/`**: Original GPD implementation (TensorFlow 1.x)
  - Contains `gpd_predict.py` - the original GPD script
  - Maintained for compatibility and reference

- **`docs/`**: Documentation and instructions
- **`env/`**: Conda environment lock files

### Model Architecture

GPD uses a deep convolutional neural network with:
- Input: 3-component seismic waveforms (400 samples = 4 seconds at 100 Hz)
- Architecture: Conv1D layers with dropout and max pooling
- Output: P-wave and S-wave probability streams
- Models require specific preprocessing: 100 Hz sampling rate, bandpass filtering (3-20 Hz)

### Key Processing Parameters

- **Sampling rate**: 100 Hz (required for model compatibility)
- **Window size**: 4 seconds (400 samples)
- **Sliding window step**: 10 samples (0.1 seconds)
- **Frequency filter**: 3-20 Hz bandpass
- **Default thresholds**: P-wave 0.95, S-wave 0.95 (configurable)

## Development Workflow

1. **Environment Setup**: Copy `.env.example` to `.env` and configure data paths
2. **Data Preparation**: Use `scripts/mseed/` tools to preprocess seismic data
3. **Model Conversion**: Use `scripts/conversion_modelos/` for format conversions if needed
4. **Inference**: Run detection using `scripts/inference/` scripts
5. **Validation**: Compare results using `scripts/debug/` analysis tools
6. **Interactive Analysis**: Use `scripts/gui/` tools for data inspection

### Typical Research Workflow

```bash
# 1. Set up environment configuration
cp .env.example .env
# Edit .env with your data paths

# Test configuration (Claude Code can run this)
python3 scripts/utils/test_config.py

# 2. Activate appropriate environment and generate dataset
micromamba activate gpd_py39
python scripts/mseed/mseed_for_gpd.py -I dataset.csv -O output_dir --input-freq 64 -V

# 3. Run batch inference with custom thresholds (uses .env config automatically)
python scripts/inference/gpd_keras_inference_events.py --min-proba-p 0.55 --min-proba-s 0.85 -V

# 4. Compare results with analyst picks
python scripts/debug/compare_analyst-automated_events.py --analyst analyst_picks.csv --automated results.csv --tolerance 0.5 1.0 2.0 5.0 -V
```

## Important Notes

- All seismic data processing assumes 3-component (North, East, Vertical) recordings
- Model expects specific preprocessing: 100 Hz sampling, 3-20 Hz filtering
- Large files are processed in chunks to manage memory usage
- Multiple model formats supported: Keras (.h5, .keras), TensorFlow SavedModel, TFLite
- Scripts include extensive logging with `-V` (verbose) flag for debugging
- **Environment configuration**: Scripts automatically load `.env` file for path configuration
- **Model selection**: Use `--model-path` with just the model name (e.g., `gpd_v1.keras`), full path is resolved automatically
- **Claude Code limitations**: Can only execute `scripts/utils/` scripts directly; other scripts require conda environments