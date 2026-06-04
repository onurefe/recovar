# RECOVAR

[![arXiv](https://img.shields.io/badge/arXiv-2407.18402v1-b31b1b.svg)](https://arxiv.org/abs/2407.18402)

## Overview

**RECOVAR** is an unsupervised machine learning framework for detecting seismic signals from continuous waveform data. It uses representation learning through deep autoencoders to distinguish seismic signals from noise without supervision, achieving competitive performance against state-of-the-art supervised methods in cross-dataset scenarios.

---

## Table of Contents

- [Installation](#installation)
- [Model Training and Testing](#model-training-and-testing)
- [SeisComP Integration](#seiscomp-integration)
- [Reproducing the Results](#reproducing-the-results)
- [License](#license)
- [Contact](#contact)

---

## Installation

**Requirements:** Python 3.10, `numpy<2.0`

### Without GPU

```bash
conda create -n recovar python=3.10
conda activate recovar
git clone git@github.com:onurefe/recovar.git
cd recovar
pip install -e .
```

### With GPU (CUDA)

```bash
conda create -n recovar python=3.10
conda activate recovar
pip install tensorflow[and-cuda]==2.14
git clone git@github.com:onurefe/recovar.git
cd recovar
pip install -e .
```

> **Note:** `numpy<2.0` is pinned in `setup.py`. Some packages can pull in numpy 2.x if installed separately before recovar — always install recovar first or pin numpy explicitly.

---

## Model Training and Testing

The root-level notebooks `model_train.ipynb` and `model_test.ipynb` provide the quickest way to train and evaluate RECOVAR on your own data. Sample data (1280 waveforms, shape `(N, 3000, 3)`) is provided in `data/` and pre-trained weights are in `models/`.

### Data format

Input waveforms are numpy arrays of shape `(N, 3000, 3)` — N samples, 3000 time steps at 100 Hz (30 s), 3 components. Labels are 1-D arrays of shape `(N,)` with `1` for earthquake and `0` for noise.

### Training — `model_train.ipynb`

1. Set paths and parameters in the Configuration cell:

   ```python
   TRAIN_DATA_PATH = 'data/X_train_1280sample.npy'
   TEST_DATA_PATH  = 'data/X_test_1280sample.npy'
   TRAIN_LABEL_PATH = 'data/Y_train_1280sample.npy'
   TEST_LABEL_PATH  = 'data/Y_test_1280sample.npy'
   MODEL_SAVE_PATH  = 'checkpoints/representation_cross_covariances.h5'
   EPOCHS = 50
   LEARNING_RATE = 1e-3
   ```

2. Choose a representation learning model (default: `RepresentationLearningMultipleAutoencoder`):

   | Model | Notes |
   |---|---|
   | `RepresentationLearningSingleAutoencoder` | Single-channel autoencoder |
   | `RepresentationLearningDenoisingSingleAutoencoder` | Denoising variant; set `input_noise_std` and `denoising_noise_std` |
   | `RepresentationLearningMultipleAutoencoder` | Multi-channel autoencoder (recommended) |

3. Run all cells. The model trains unsupervised on `X_train` (labels are not used during training). Checkpoints are saved each epoch; early stopping on validation loss is enabled.

### Testing — `model_test.ipynb`

1. Set `TEST_DATA_PATH`, `TEST_LABEL_PATH`, and `MODEL_PATH` (defaults point to `data/` and `models/representation_cross_covariances.h5`).

2. Choose the matching classifier wrapper:

   | Representation model | Classifier wrapper |
   |---|---|
   | `RepresentationLearningSingleAutoencoder` | `ClassifierAutocovariance` or `ClassifierAugmentedAutoencoder` |
   | `RepresentationLearningDenoisingSingleAutoencoder` | `ClassifierAutocovariance` or `ClassifierAugmentedAutoencoder` |
   | `RepresentationLearningMultipleAutoencoder` | `ClassifierMultipleAutoencoder` |

3. Run all cells. The notebook outputs per-sample earthquake probabilities and plots the ROC curve with AUC.

---

### MiniSEED predictor — `mseed_predictor.py`

Score a continuous 3-component MiniSEED file without SeisComP. The script slides a window across the file and outputs an earthquake probability for each position.

**Pipeline per window:**
1. Fetch a 40-second window from each component (ZNE or Z12)
2. Drop the window if any component contains a gap
3. Resample to 100 Hz if required
4. Apply a 1–20 Hz ideal Fourier bandpass
5. Crop the inner 30 seconds (removing the 5-second filter-edge buffers)
6. Score with the RECOVAR classifier

```bash
# Print scores to stdout
python mseed_predictor.py \
    --input  waveforms.mseed \
    --model  models/representation_cross_covariances.h5 \
    --step   10

# Save to CSV
python mseed_predictor.py \
    --input  waveforms.mseed \
    --model  models/representation_cross_covariances.h5 \
    --step   10 \
    --output scores.csv
```

| Argument | Default | Description |
|---|---|---|
| `--input` / `-i` | required | MiniSEED file path |
| `--model` / `-m` | required | Model weights (`.h5`) |
| `--step` / `-s` | `10` s | Step between successive windows |
| `--output` / `-o` | stdout | CSV output path |

Output columns: `window_start` (40 s fetch window), `inner_start` (start of the scored 30 s window, = `window_start + 5 s`), `score`.

---

## SeisComP Integration

`seiscomp_integration/` contains a SeisComP daemon (`recovar_pick_filter`) that listens on the PICK messaging group, fetches waveforms for each incoming pick, and attaches a `recovar_score:[0–1]` comment to the pick object in the database. A score sweep at ±40 s offsets (5 s steps) is also stored as `recovar_score_sweep`.

Tested on **Ubuntu 22.04 + SeisComP 7.x**.

- **Installation:** [`seiscomp_integration/INSTALL.md`](seiscomp_integration/INSTALL.md)
- **Running & testing:** [`seiscomp_integration/RUN.md`](seiscomp_integration/RUN.md)

### Quick summary

```bash
# Start the daemon
seiscomp start scmaster scdb recovar_pick_filter

# Query and visualise scored picks
python3 seiscomp_integration/query_scored_picks.py
python3 seiscomp_integration/query_scored_picks.py -o picks.csv
python3 seiscomp_integration/query_scored_picks.py --plot --plot-output fig.png
```

---

## Reproducing the Results

### 1. Download datasets

Download [STEAD](https://github.com/smousavi05/STEAD) and [INSTANCE](http://repo.pi.ingv.it/instance) and note their local paths.

### 2. Configure paths

Edit `reproducibility/config.py` and set:

- `STEAD_WAVEFORMS_HDF5_PATH` / `STEAD_METADATA_CSV_PATH`
- `INSTANCE_EQ_WAVEFORMS_HDF5_PATH` / `INSTANCE_EQ_METADATA_CSV_PATH`
- `INSTANCE_NOISE_WAVEFORMS_HDF5_PATH` / `INSTANCE_NOISE_METADATA_CSV_PATH`
- `PREPROCESSED_DATASET_DIRECTORY`, `TRAINED_MODELS_DIR`, `RESULTS_DIR`

Key parameters (defaults reproduce the paper):

| Parameter | Default | Description |
|---|---|---|
| `SUBSAMPLING_FACTOR` | `1.0` | Fraction of data to use |
| `TRAIN_VALIDATION_SPLIT` | `0.75` | Train / validation ratio |
| `KFOLD_SPLITS` | `5` | Number of k-fold splits |
| `DATASET_CHUNKS` | `20` | Chunks for k-fold preprocessing |

### 3. Train

Open `reproducibility/training.ipynb`. Initial preprocessing takes a few hours; full training (5-fold, three models, both datasets, 20 epochs) takes approximately one day on an NVIDIA RTX 3090 Ti.

### 4. Test

Open `reproducibility/testing.ipynb`. Uses `kfold_tester` to generate earthquake probabilities and `evaluator` to compute metrics.

---

## License

This project is licensed under the MIT License.

## Contact

For questions, issues, or feature requests, open an issue on GitHub or contact onur.efe44@gmail.com.
