import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

sp500 = pd.read_csv('sphist.csv')
sp500['Date'] = pd.to_datetime(sp500['Date'])

sp500 = sp500.sort_values(by = 'Date',
                          ascending = True)
sp500.reset_index(drop = True)

# calculating the mean of the past 5/10/365 days:
sp500['mean_5'] = sp500['Close'].rolling(5).mean().shift(1)
sp500['mean_10'] = sp500['Close'].rolling(10).mean().shift(1)
sp500['mean_365'] = sp500['Close'].rolling(365).mean().shift(1)
sp500['std_5'] = sp500['Close'].rolling(5).std().shift(1)
sp500['std_10'] = sp500['Close'].rolling(10).std().shift(1)
sp500['std_365'] = sp500['Close'].rolling(365).std().shift(1)

sp500['mean_5/mean_365'] = sp500['mean_5']/sp500['mean_365']
sp500['std_5/std_365'] = sp500['std_5']/sp500['std_365']

sp500_copy = sp500.copy()

#before splitting into test and train set: dropping the rows that hadn't enough historical data to compute the means above
date_bool = sp500["Date"] > datetime(year=1951, month=1, day=3)
sp500 = sp500[date_bool]
sp500.dropna(axis = 0,inplace = True)

#splitting into test and train set:
test_train_bool = sp500["Date"] > datetime(year=2013, month=1, day=1)
train = sp500[~test_train_bool].copy()
test = sp500[test_train_bool].copy()

###initialising linear regression:
lr = LinearRegression()
not_train_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date']
train_col = train.columns.drop(not_train_cols)
target = 'Close'
lr.fit(train[train_col],train[target])
predictions = lr.predict(test[train_col])
rmse = np.sqrt(mean_squared_error(predictions, test[target]))
print('rmse all columns: ',rmse)

#Now I will test whether it was a good idea to keep the columns with ratios in the train set:
lr = LinearRegression()
not_train_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date','mean_5/mean_365','std_5/std_365']
train_col = train.columns.drop(not_train_cols)
target = 'Close'
lr.fit(train[train_col],train[target])
predictions = lr.predict(test[train_col])
rmse = np.sqrt(mean_squared_error(predictions, test[target]))
print('rmse selected columns: ',rmse)

#Now I will test whether some stats about "Volume" will improve the model:
sp500_copy['mean_5_vol'] = sp500_copy['Volume'].rolling(5).mean().shift(1)
sp500_copy['mean_10_vol'] = sp500_copy['Volume'].rolling(10).mean().shift(1)
sp500_copy['mean_365_vol'] = sp500_copy['Volume'].rolling(365).mean().shift(1)
sp500_copy['std_5_vol'] = sp500_copy['Volume'].rolling(5).std().shift(1)
sp500_copy['std_10_vol'] = sp500_copy['Volume'].rolling(10).std().shift(1)
sp500_copy['std_365_vol'] = sp500_copy['Volume'].rolling(365).std().shift(1)

date_bool = sp500_copy["Date"] > datetime(year=1951, month=1, day=3)
sp500_copy = sp500_copy[date_bool]
sp500_copy.dropna(axis = 0,inplace = True)

test_train_bool = sp500_copy["Date"] > datetime(year=2013, month=1, day=1)
train = sp500_copy[~test_train_bool].copy()
test = sp500_copy[test_train_bool].copy()

lr = LinearRegression()
not_train_cols = ['Close', 'High', 'Low', 'Open', 'Volume', 'Adj Close', 'Date','mean_5/mean_365','std_5/std_365']
train_col = train.columns.drop(not_train_cols)
target = 'Close'
lr.fit(train[train_col],train[target])
predictions = lr.predict(test[train_col])
rmse = np.sqrt(mean_squared_error(predictions, test[target]))
print('rmse selected columns + Volume: ',rmse)
