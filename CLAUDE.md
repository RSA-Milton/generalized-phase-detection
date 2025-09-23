# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research repository for Generalized Phase Detection (GPD), a deep learning framework for seismic phase detection developed by Ross et al. (2018). The codebase contains Python scripts for model conversion, inference, validation, and analysis of seismic data.

## Environment Setup

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

## Common Commands

### Data Preprocessing
```bash
# Preprocess individual mseed file for GPD
python scripts/mseed/preprocess_mseed_for_gpd.py -I input.mseed -O output.mseed --input-freq 64 -V

# Batch processing of directory
python scripts/mseed/preprocess_mseed_for_gpd.py -I input_dir/ -O output_dir/ --input-freq 64 --batch -V

# Generate dataset with SNR information
python scripts/mseed/mseed_for_gpd.py -I dataset.csv -O output_dir --input-freq 64 -V
```

### Inference
```bash
# Keras inference (primary method)
python validacion/inferencia_keras.py -I input.mseed -O output.out -V

# Legacy inference (TensorFlow 1.x)
python validacion/inferencia_legacy.py -I input.mseed -O output.out -V --hours 4

# STA/LTA detection
python validacion/stalta_detector.py -I input.mseed -O output.out -V
python validacion/stalta_detector.py -I input.mseed -O output.out --coincidence 2 -V

# Batch evaluation with custom thresholds
python scripts/inference/evaluate_gpd_events.py --min-proba-p 0.55 --min-proba-s 0.85 -V
```

### Analysis and Comparison
```bash
# Compare GPD results with analyst picks
python scripts/utils/compare_analyst_gpd_enhanced.py --analyst analyst_picks.csv --gpd gpd_results.csv --tolerance 0.5 1.0 2.0 5.0 -V

# Evaluate detection performance
python validacion/evaluate_gpd_events.py --min-proba-p 0.50 --min-proba-s 0.85 -V
```

## Architecture

### Core Directories

- **`scripts/`**: Main processing scripts organized by function
  - `conversion_modelos/`: Model format conversion tools (HDF5, SavedModel, TFLite)
  - `inference/`: Batch inference and evaluation scripts
  - `mseed/`: MSEED data processing and preprocessing
  - `utils/`: Analysis and comparison utilities
  - `gui/`: Interactive data inspection tools

- **`validacion/`**: Validation and testing scripts
  - Different inference implementations (Keras, legacy, TFLite)
  - Performance comparison tools
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

1. **Data Preparation**: Use `scripts/mseed/` tools to preprocess seismic data
2. **Model Conversion**: Use `scripts/conversion_modelos/` for format conversions
3. **Inference**: Run detection using `validacion/` or `scripts/inference/` scripts
4. **Validation**: Compare results using `scripts/utils/` analysis tools
5. **Debugging**: Use `scripts/debug/` tools for troubleshooting

## Important Notes

- All seismic data processing assumes 3-component (North, East, Vertical) recordings
- Model expects specific preprocessing: 100 Hz sampling, 3-20 Hz filtering
- Large files are processed in chunks to manage memory usage
- Multiple model formats supported: Keras (.h5), TensorFlow SavedModel, TFLite
- Scripts include extensive logging with `-V` (verbose) flag for debugging