import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta

class StockPredictor:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.model = LinearRegression()
        self.data = None
        self.future_data = None

    def download_data(self):
        # Download stock data using yfinance
        self.data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        self.data = self.data[['Close']]
        # Convert the dates to ordinal values for linear regression
        self.data['Date_ordinal'] = pd.to_datetime(self.data.index).map(lambda date: date.toordinal())

    def build_model(self):
        # Prepare the data for linear regression
        X = self.data['Date_ordinal'].values.reshape(-1, 1)
        y = self.data['Close'].values
        # Fit the model to the historical data
        self.model.fit(X, y)
        # Predict the trend and add it to the data
        self.data['Trend'] = self.model.predict(X)

    def predict_future(self, days=365):
        # Predict stock prices for the future (e.g., next 365 days)
        future_dates = pd.date_range(start=self.data.index[-1] + timedelta(days=1), periods=days, freq='D')
        future_dates_ordinal = future_dates.map(lambda date: date.toordinal()).values.reshape(-1, 1)
        future_predictions = self.model.predict(future_dates_ordinal)
        
        # Create a DataFrame for future predictions
        self.future_data = pd.DataFrame({
            'Date': future_dates,
            'Predicted_Close': future_predictions
        })

    def plot_predictions(self):
        # Plot the actual stock prices, the trend, and future predictions
        plt.figure(figsize=(12, 6))

        # Plot actual stock prices
        plt.plot(self.data.index, self.data['Close'], label='Actual Stock Prices', marker='o', markersize=3)

        # Plot linear regression trend
        plt.plot(self.data.index, self.data['Trend'], label='Linear Regression Trend', linestyle='--')

        # Plot future predictions
        if self.future_data is not None:
            plt.plot(self.future_data['Date'], self.future_data['Predicted_Close'], label='Future Predictions', linestyle='-.', color='red')

        # Plot settings
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title(f'Trend and Future Predictions for {self.symbol} Using Linear Regression')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Show the plot
        plt.show()

# Usage example:

# Create a StockPredictor object
predictor = StockPredictor('AAPL', '2020-01-01', '2023-01-01')

# Download historical stock data
predictor.download_data()

# Build the linear regression model
predictor.build_model()

# Predict stock prices for the next year (365 days)
predictor.predict_future(days=365)

# Plot the actual prices, trend, and future predictions
predictor.plot_predictions()
