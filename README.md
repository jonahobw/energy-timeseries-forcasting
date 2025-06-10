# Energy Time Series Forecasting

Author: [Jonah O'Brien Weiss](https://jonahobw.github.io)

Implemented on: 6/9/2025

This repo contains a simple demonstration of timeseries forcasting in both PyTorch and TensorFlow on the the [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) dataset.  It was implemented as a learning exercise.

\*\*Disclaimer\*\* - Parts of this code were generated using Cursor, but I have validated the output.

## Dataset

This project uses the [ElectricityLoadDiagrams20112014](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) dataset from the UCI Machine Learning Repository. The dataset contains electricity consumption data from 370 clients, with measurements taken every 15 minutes from 2011 to 2014.

Dataset characteristics:
- Time series data
- 370 clients
- 140,256 time steps per client
- Values in kW
- No missing values
- Portuguese time zone

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Download the dataset:
```bash
python download_electricity.py
```

## Project Structure

The project contains several Python modules for time series forecasting:

### `data_processing.py`
- Handles data loading and preprocessing
- Implements standardization methods (mean/std and IQR)
- Provides visualization utilities for predictions

### `timeseries_forcasting_tf.py`
- TensorFlow implementation of LSTM-based time series forecasting
- Features:
  - Custom data generator for efficient batch processing
  - Simple LSTM model with configurable architecture
  - Training and validation pipeline
  - Prediction visualization

### `timeseries_forcasting_torch.py`
- PyTorch implementation of the same forecasting model
- Useful for comparing frameworks

## Usage

To train and evaluate the TensorFlow model:
```bash
python timeseries_forcasting_tf.py
```

To train and evaluate the PyTorch model:
```bash
python timeseries_forcasting_torch.py
```

Both implementations will:
1. Load and preprocess the data
2. Train an LSTM model
3. Generate predictions
4. Display training curves and prediction visualizations

## Model Architecture

The models use a simple LSTM architecture:
- Input: Time series window of configurable size
- LSTM layer with configurable hidden size
- Dense output layer for prediction
- MSE loss function
- Adam optimizer