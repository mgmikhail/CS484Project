# CS484Project: ML Stock Prediction Tool

## Description
This project explores the use of Recurrent Neural Networks (RNNs)help predict the flow of the stock market, and help investors make smart decisions. It features several models: <br>
<ul>
  <li>A baseline LSTM model</li>
  <li>An LSTM + EMA (Exponential Moving Average) model</li>  
  <li>Two volatility-sensitive models, incorporating custom metrics in an attempt to omprove predictions during highly volatile periods.</li>
</ul>
The results highlight both the potential and the limitations of using machine learning for stock market prediction. While general trends can be modeled, the inherently unpredictable nature of the stock market,especially during volatile periods, is challenging.<br>

## Files
[lstm1.1](lstm1.1.py): basic LSTM model with no additional features (S&P) <br>
[lstm1](lstm1.py): LSTM model with EMA as a parameter (S&P)<br>
[lstm1Bitcoin](lstm1Bitcoin.py): LSTM model with EMA, set up to preprocess Bitcoin Historical Data<br>
[lstm1withVolatility](lstm1withVolatility.py): LSTM model with EMA and volatility metric (S&P)<br>
[lstm1withVolatilityBitcoin](lstm1withVolatilityBitcoin.py): LSTM model with EMA and volatility metric for Bitcoin Data<br>
[HistoricalData_1741813699330](HistoricalData_1741813699330.csv): S&P historical data<br>
[BitcoinHistoricalData](BitcoinHistoricalData.csv): Bitcoin Historical data<br><br>

*Some notes to keep in mind for running programs:*<br>
All the programs take data from the filepath which is hardcoded in the very beginning of the program. In order to run on your machine, download the csv files provided and change the filepath variable in your IDE. All programs can hypothetically run with any similar dataset, however the data will need to be processed differently. Note that for all programs, the current setup has 40 epochs, which we found ideal for output, however it will take a minimum of 1-2 minutes to run each program. If you need to run the program quickly and accuracy of output is not as important, adjust the epochs down to 20 or 15. Lower than 10 epochs is not recomended, outputs will be all but useless.<br>
<br>

<h2>Output</h2>
<ul>
  <li>Graph showing predicted data vs. test data</li>
  <li>MSE, MAE, and R^2 values</li>
</ul>
<br>

<h2>Dependencies:</h2>
Dependencies can be installed using the following line:
<code>pip install pandas scikit-learn numpy tensorflow joblib matplotlib<code\>
