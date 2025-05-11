# Weather-Temperature-Prediction-using-LSTM
## Project Overview

This project is part of the **Natural Language Processing (NLP) course** at **Cyprus International University**. The goal is to predict weather temperatures using **Long Short-Term Memory (LSTM)** networks. 

## Technologies Used

- **TensorFlow** and **Keras** for building and training the LSTM model.
- **Pandas** and **NumPy** for handling data.

## Model Architecture

The model is built using **Long Short-Term Memory (LSTM)** layers, which are designed to capture temporal dependencies in time series data. The architecture consists of multiple LSTM layers followed by dropout layers to prevent overfitting and a dense layer for final prediction.

- **Input Shape**: (144, 1) — Each input sample consists of 144 time steps, representing 24 hours of weather data at 10-minute intervals.
- **LSTM Layers**: 128 units, 64 units, 32 units, and 16 units.
- **Dropout**: To avoid overfitting, dropout layers are added after each LSTM layer.
- **Activation Function**: tanh for LSTM layers.
- **Final Layer**: Dense layer with a single output unit for temperature prediction.

## Dataset

The dataset used in this project is the [Weather Long-term Time Series Forecasting](https://www.kaggle.com/datasets/alistairking/weather-long-term-time-series-forecasting) dataset available on Kaggle. It contains weather data recorded every 10 minutes throughout the entire year of 2020, comprising 20 meteorological indicators measured at a Max Planck research site. Each data point represents a 10-minute interval, and for this project, sequences of 144 consecutive readings (representing 24 hours) are used as input for the LSTM model.


## Data Preprocessing

- **Standardization**: The data was standardized using **StandardScaler**.
- **Sliding Window**: The time series data was split into sequences of 144 time steps (representing 24 hours) using a sliding window approach.
- **Train-Test Split**: The data was divided into training and test sets for model evaluation.


## Model Training

## Training Details

- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Optimizer**: Adam with a learning rate of 0.001
- **Epochs**: 50
- **Batch Size**: 32
  
The model was trained using a validation split and an **EarlyStopping** callback to prevent overfitting. The model achieved an **R² score of 0.9949**, indicating it explains over 99.4% of the variance in the data.

### Training Results:

- **Training Loss**: Reduced from 0.0648 to 0.0185 across epochs.
- **Validation Loss**: Fluctuated but eventually stabilized after several epochs.
- **R² Score**: 0.9949, indicating that the model explains over 99.4% of the variance in the data.

Despite occasional fluctuations in validation loss, the model demonstrated strong performance overall. **EarlyStopping** was used to restore the model to the best-performing epoch.
