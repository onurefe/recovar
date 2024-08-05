# The Seismic Purifier

[![arXiv](https://img.shields.io/badge/arXiv-2407.18402v1-b31b1b.svg)](https://arxiv.org/abs/2407.18402)

## Overview

**The Seismic Purifier** is an unsupervised machine learning framework for detecting seismic signals from continuous waveform data. By leveraging representation learning through deep auto-encoders, this method aims to effectively distinguish between seismic signals and noise without supervision, offering competitive performances to many state-of-the-art supervised methods in cross-dataset scenarios.

## Features
- **Unsupervised Learning**: Utilizes deep auto-encoders to learn compressed representations of seismic waveforms, requiring minimal labeled data.
- **Robust Performance**: Demonstrates superior detection capabilities compared to existing supervised methods, with strong cross-dataset generalization.
- **Scalability**: Designed to handle large-scale time-series data, making it applicable to various signal detection tasks beyond seismology.
- **Intuitive Design**: Employs a time-axis-preserving approach and a straightforward triggering mechanism to differentiate noise from signals.
