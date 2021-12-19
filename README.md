# EECS595-FINAL
FINAL PROJECT FOR EECS595

## Stock Selection Alpha-Factor for Chinese Stock Market Based on Sentiment Analysis

The code and data of the final project for EECS595 is stored at here.

The environment I used is python 3.8 and pytorch 1.10 with RTX3080 GPU

###  Introduction

* "Stock_Data" is the folder that contains the original stock data that exported from Wind Finance Terminal.
* "Stock_data_c" is the folder that contains the cleaned stock data and the calculation result of the return for each stock.
* "dataset" is the folder that contains the crawled raw news data from the webset.
* "data_cleaned" folder contains the labeled raw news data
* Notebook "GetData" is the python crawler
* Notebook "DataProcess1" is the preprocss of the raw news data
* "BERT.py" is the file that train and evaluate the BERT model, use "python BERT.py" to run this file in terminal and get the train and validation result of BERT model.
* "LSTMmodel.py" is the file that contains train, evaluate and predict of the LSTM model and the bscktesting framework of the project. Uss "python LSTMmodel.py" in terminal to run this file.
