# CS484Project

Files Description:
lstm1.1: basic LSTM model with no additional features (S&P)
lstm1: LSTM model with EMA as a parameter (S&P)
lstm1Bitcoin: LSTM model with EMA, set up to preprocess Bitcoin Historical Data
lstm1withVolatility: LSTM model with EMA and volatility metric (S&P)
lstm1withVolatilityBitcoin: LSTM model with EMA and volatility metric for Bitcoin Data
HistoricalData_1741813699330: S&P historical data
BitcoinHistoricalData: Bitcoin Historical data

Some notes to keep in mind for running programs:
All the programs take data from the filepath which is hardcoded in the very beginning of the program. In order to run on your machine, download the csv files provided and change the filepath variable in your IDE. All programs can hypothetically run with any similar dataset, however the data will need to be processed differently. Note that for all programs, the current setup has 40 epochs, which we found ideal for output, however it will take a minimum of 1-2 minutes to run each program. If you need to run the program quickly and accuracy of output is not as important, adjust the epochs down to 20 or 15. Lower than 10 epochs is not recomended, outputs will be all but useless.
Output:
- graph showing predicted data vs. test data
- MSE, MAE, and R^2 values


Dependencies:
Dependencies can be installed using the following line:
pip install pandas scikit-learn numpy tensorflow joblib matplotlib
