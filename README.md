# Weather-Temperature-Prediction-using-LSTM
## Project Overview

This project is part of the **Natural Language Processing (NLP) course** at **Cyprus International University**. The goal is to predict weather temperatures using **Long Short-Term Memory (LSTM)** networks. 

## Technologies Used

- **TensorFlow** and **Keras** for building and training the LSTM model.
- **Pandas** and **NumPy** for handling data.
- **Scikit-learn** for scaling and performance evaluation

## Model Architecture

The model is built using **Long Short-Term Memory (LSTM)** layers, which are designed to capture temporal dependencies in time series data. The architecture consists of multiple LSTM layers followed by dropout layers to prevent overfitting and a dense layer for final prediction.

- **Input Shape**: (144, 1) — Each input sample consists of 144 time steps, representing 24 hours of weather data at 10-minute intervals.
- **LSTM Layers**: 128 units, 64 units, 32 units, and 16 units.
- **Dropout**:Added after each LSTM layer to reduce overfitting (rates: 0.3 → 0.3 → 0.3 → 0.2).
- **Activation Function**: tanh for LSTM layers.
- **Layer Normalization**: Added after the final LSTM layer to improve convergence stability.
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
- **Epochs**: Up to 50, with EarlyStopping (patience = 5) to avoid overfitting.
- **Batch Size**: 32
  
The model was trained using a validation split and an **EarlyStopping** callback to prevent overfitting. The model achieved an **R² score of 0.9949**, indicating it explains over 99.4% of the variance in the data.

### Training Results:
The model achieved strong generalization performance, with validation loss stabilizing after a few epochs and test performance indicating excellent prediction accuracy.

Final Metrics:
Best Validation Loss: 0.0057
Best Validation MAE: 0.0622
Test R² Score: 0.9949 — the model explains over 99.4% of the variance in the temperature data.

This project demonstrates that a well-tuned LSTM model can effectively forecast temperature with high accuracy using high-frequency weather data. Incorporating deeper LSTM layers and normalization significantly improved stability and generalization.
