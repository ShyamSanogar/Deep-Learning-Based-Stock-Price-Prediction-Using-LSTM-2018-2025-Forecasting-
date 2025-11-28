# Deep-Learning-Based-Stock-Price-Prediction-Using-LSTM-2018-2025-Forecasting-
This project builds a deep learning model using LSTM (Long Short-Term Memory networks) to predict stock prices based on historical market data and technical indicators. It includes an end-to-end pipeline for data collection, feature engineering, model training, backtesting, and multi-step future forecasting up to 2025.
🚀 Project Overview

The goal of this project is to forecast future stock price movements using LSTM neural networks while incorporating technical indicators such as RSI, SMA, MACD, and Bollinger Bands to improve predictive performance.

The model fetches real-time historical data using Yahoo Finance, preprocesses it, generates sequences, and predicts both test performance and future values.

📊 Key Features
🔹 1. Data Collection

Fetches historical stock data (2018 → Today) via yfinance

Supports user-input ticker symbols

Auto-handling of rate limits with retry logic

🔹 2. Technical Indicator Engineering (TA-Lib)

Adds financial indicators:

SMA 50 & SMA 200

RSI (14)

MACD & Signal Line

Bollinger Bands (Upper, Middle, Lower)

🔹 3. Data Preprocessing

Cleaning and NaN handling

Feature scaling using MinMaxScaler

Windowed sequence creation for LSTM

Train-test split (80–20)

🔹 4. LSTM Model Architecture

Stacked LSTM layers (50 units each)

Dropout regularization

Dense layers for regression

Trained for 1000 epochs

🔹 5. Predictions & Evaluation

Predicts closing prices on test data

Evaluation metrics:

MAE (Mean Absolute Error)

MSE (Mean Squared Error)

🔹 6. Future Forecasting (2025)

Predicts stock prices up to Dec 2025

Uses business-day frequency

Rolling-window prediction logic

🔹 7. Visualization

Plots:

Actual vs. Predicted

Future forecast (dashed line)

All graphs via Matplotlib

🧠 Model Workflow

Fetch data

Compute technical indicators

Normalize features

Create time sequences

Train LSTM model

Predict test data

Forecast future data

Visualize trends

📂 Project Structure (Recommended for GitHub)
📦 Stock-Price-Prediction-LSTM
├── 📁 data/                # Optional saved datasets
├── 📁 models/              # Saved trained models
├── 📁 plots/               # Exported charts
├── stock_predictor.py     # Main project script
├── requirements.txt        # Required libraries
└── README.md               # Documentation

🛠️ Tech Stack Used

Languages & Frameworks:

Python

TensorFlow / Keras

Scikit-learn

Libraries:

NumPy

Pandas

TA-Lib

Matplotlib

yfinance

Techniques:

LSTM networks

Time-series forecasting

Technical indicators

Sliding window prediction

📥 Installation
1️⃣ Clone the repo
git clone https://github.com/your-username/Stock-Price-Prediction-LSTM.git
cd Stock-Price-Prediction-LSTM

2️⃣ Install dependencies
pip install -r requirements.txt

▶️ Usage

Run the main script:

python stock_predictor.py


Enter your stock ticker when prompted:

Enter the stock name: TCS.NS
Past days data to be checked: 50


The script will:
✔ fetch the data
✔ compute indicators
✔ train LSTM
✔ predict test prices
✔ forecast until 2025
✔ display full chart

📉 Output Example

(You can add your plot images here in the repository)

Mean Absolute Error: <value>
Mean Squared Error: <value>


Graph includes:

Actual price (blue)

Test prediction (red)

Forecast (green dashed)

📌 Future Improvements

Add GRU- or Transformer-based models

Incorporate volume and sentiment analysis

Build a Streamlit dashboard

Hyperparameter tuning with Optuna
