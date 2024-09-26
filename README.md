# ReCovar

[![arXiv](https://img.shields.io/badge/arXiv-2407.18402v1-b31b1b.svg)](https://arxiv.org/abs/2407.18402)

## Overview

**ReCovar** is an unsupervised machine learning framework for detecting seismic signals from continuous waveform data. By leveraging representation learning through deep auto-encoders, this method aims to effectively distinguish between seismic signals and noise without supervision, offering competitive performances to many state-of-the-art supervised methods in cross-dataset scenarios.

## Features
- **Unsupervised Learning**: Utilizes deep auto-encoders to learn compressed representations of seismic waveforms, requiring minimal labeled data.
- **Robust Performance**: Demonstrates superior detection capabilities compared to existing supervised methods, with strong cross-dataset generalization.
- **Scalability**: Designed to handle large-scale time-series data, making it applicable to various signal detection tasks beyond seismology.
- **Intuitive Design**: Employs a time-axis-preserving approach and a straightforward triggering mechanism to differentiate noise from signals.

## Table of Contents

- [SeismicPurifier:](#symmetry-lens)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Using conda (Without GPU)](#using-conda-without-gpu)
    - [Using conda (With GPU)](#using-conda-with-gpu)
    - [Direct installation (Without GPU)](#direct-installation-without-gpu)
    - [Direct installation (With GPU)](#direct-installation-with-gpu)
  - [Reproducing the Results](#reproducing-the-results)
   - [Training the models](#training-the-models)
   - [Testing the models](#testing-the-models)
  - [Experimenting with Custom Data](#experimenting-with-custom-data)
  - [License](#license)
  - [Contact](#contact)

---

## Installation

### Prerequisites

- **Python Version:** Ensure you are using **Python 3.10**.
- **NVIDIA GPU Drivers (If using GPU):** Required for GPU support.
- **CUDA and cuDNN Libraries (If using GPU):** Compatible versions for TensorFlow 2.14.0.

### Using conda (Without GPU)
- **Create and activate conda environment**
   ```bash
   conda create -n <environment_name> python=3.10
   conda activate <environment_name>
   ```
- **Install package SeismicPurifier**
   ```bash
   git clone git@github.com:onurefe/SeismicPurifier.git
   cd SeismicPurifier
   python setup.py install
   ```

### Using conda (With GPU)
- **Create and activate conda environment**
   ```bash
   conda create -n <environment_name> python=3.10
   conda activate <environment_name>
   ```

- **Install tensorflow and cuda/cudnn libraries**
   ```bash
   pip install tensorflow[and-cuda]==2.14
   ```

- **Install package SeismicPurifier**
   ```bash
   git clone git@github.com:onurefe/SeismicPurifier.git
   cd SeismicPurifier
   python setup.py install
   ```

### Direct installation (Without GPU)
- **Install package SeismicPurifier**
   ```bash
   git clone git@github.com:onurefe/SeismicPurifier.git
   cd SeismicPurifier
   python setup.py install
   ```

### Direct installation (With GPU)
- **Install tensorflow and cuda/cudnn libraries**
   ```bash
   pip install tensorflow[and-cuda]==2.14
   ```

- **Install package SeismicPurifier**
   ```bash
   git clone git@github.com:onurefe/SeismicPurifier.git
   cd SeismicPurifier
   python setup.py install
   ```

## Reproducing the Results
### Downloading dataset
In order to reproduce the results given in the paper, first you need to download stead and instance datasets. Visit [STEAD](https://github.com/smousavi05/STEAD) and [INSTANCE](http://repo.pi.ingv.it/instance) for downloading instructions.

### Adjusting paths, and settings
After downloading the dataset, we need to configure **settings.json** file in the SeismicPurifier/reproducibility folder. 
This file provides path variables and there are multiple experimentation options which you may consider adjusting for different purposes. 

#### Configurable parameters
For reproducing the current results, keep these values as they are.
   - **CONFIG**
      - **SUBSAMPLING_FACTOR**: Default value 1.0. Change this factor if you want to use less data for training, testing and validation.

      - **TRAIN_VALIDATION_SPLIT**: Default value 0.75. The ratio of training set size to the validation set size.

      - **KFOLD_SPLITS**: Default value 5. Dataset is split into equal parts for obtaining statistics about the performance of the model. This parameter controls the number of splits.

      - **DATASET_CHUNKS**: Default value 20. We split the dataset into chunk of smaller portions before using kfold validation which boosts the training performance.

      - **PHASE_PICK_ENSURED_CROP_RATIO**: Default value 0.666666. During the training period, if the dataset window size is longer than 30s, we crop the waveform randomly. However, this crop may or may not include the P arrival event. This ratio narrows down the selection of crop window positions such that phase arrival event is ensured to be included for %66(for the default value) of the waveform samples.

      - **PHASE_ENSURING_MARGIN**: Marging of the definition of including the P arrival event. 

#### Directories
Adjust paths of the datasets after downloading them to your system.
   - **DATASET_DIRECTORIES**
      - **STEAD_WAVEFORMS_HDF5_PATH**: Path to the STEAD dataset's waveforms stored in HDF5 format. Ensure this path points to the correct location of your STEAD waveforms file.

      - **STEAD_METADATA_CSV_PATH**: Path to the STEAD dataset's metadata stored in CSV format. This file contains essential metadata associated with the STEAD waveforms.
      
      - **INSTANCE_NOISE_WAVEFORMS_HDF5_PATH**: Path to the INSTANCE dataset's noise waveforms stored in HDF5 format. Ensure this path points to the correct location of your INSTANCE noise waveforms file.

      - **INSTANCE_EQ_WAVEFORMS_HDF5_PATH**: Path to the INSTANCE dataset's earthquake waveforms stored in HDF5 format. Ensure this path points to the correct location of your INSTANCE earthquake waveforms file.

      - **INSTANCE_NOISE_METADATA_CSV_PATH**: Path to the INSTANCE dataset's noise metadata stored in CSV format. This file contains essential metadata associated with the INSTANCE noise waveforms.

      - **INSTANCE_EQ_METADATA_CSV_PATH**: Path to the INSTANCE dataset's earthquake metadata stored in CSV format. This file contains essential metadata associated with the INSTANCE earthquake waveforms.
    
   - **PREPROCESSED_DATASET_DIRECTORY**: Directory where preprocessed datasets will be stored. Ensure this directory exists or the application has permissions to create it.

   - **TRAINED_MODELS_DIR**: Directory to store trained machine learning models. Ensure this directory exists or the application has permissions to create it.
    
   - **RESULTS_DIR**: Directory to store the classification results. 

### Training the models
After completing the download and setting adjustment procedures, proceed to the **reproducibility** folder inside the SeismicPurifier 
directory.

By using **training.ipynb** you can train all available of models on both datasets. At the initial phase of the training, datapreprocessing part may take a while(approximately couple of hours). However, it boost the training procedure significantly. On NVIDIA RTX3090 Ti, all training procedure(5-Fold, three models, whole of two datasets and for 20 epochs) takes approximately a day. 

### Testing the models
After the training the models, you can test different method performances by using kfold_tester(for obtaining unnormalized earthquake probabilities) and evaluator. **testing.ipynb** provides a template for the testing procedure. 

## Experimenting with Custom Data.
If your dataset is compatible with the structure of either INSTANCE or STEAD datasets, you can use all machinery under the folder **reproducibility**. Or, a different option could be converting your data
into STEAD dataset format by using [QuakeLabeler](https://maihao14.github.io/QuakeLabeler/) or [SeisBench](https://github.com/seisbench/seisbench).

For other types of data, it's possible to feed numpy arrays directly for training. **SeismicPurifier/model_train.ipynb** provides example for this case. You can also test your data as well. Please 
check **SeismicPurifier/model_test.ipynb**. For training and testing, pretrained models are stored in **SeismicPurifier/models** folder. Besides, **SeismicPurifier/data** involves small dataset
for experimenting. 

## License
This project is licensed under the MIT License.

## Contact
For any questions, issues, or feature requests, please open an issue on the GitHub repository contact onur.efe44@gmail.com.
