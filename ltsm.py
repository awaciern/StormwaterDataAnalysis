# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM

# Register converters
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# Read in the big dataset with flood data
data = pd.read_csv('../Data/Big/Flood.csv', index_col='Unnamed: 0')
print(data)

# Feature to predict (CHANGE TO PREDICT DIFFERENT THINGS)
feat = 'B_Flow'

# Create a new dataset with just the X flood data
new_data = pd.DataFrame(index=range(0, len(data)), columns=['Date', feat])
for i in range(0, len(data)):
    new_data['Date'][i] = data.index[i]
    new_data[feat][i] = data[feat][i]
new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)
print(new_data)

# Create train and test data from this new dataset
data_vals = new_data.values
train = data_vals[0:17569, :]
test = data_vals[17569:, :]
print(test)
print(train)

# Create scaled training data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_vals)
x_train, y_train = [], []
for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
print(x_train)
print(y_train)

# Reshape x training data for LTSM network
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
print(x_train)

# Create and fit the LTSM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=1)

# Predict values
inputs = new_data[len(new_data) - len(test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)
x_test = []
for i in range(60,inputs.shape[0]):
    x_test.append(inputs[i-60:i,0])
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
preds = model.predict(x_test)
preds = scaler.inverse_transform(preds)

# Calculate and print MSE
mse = np.mean(np.power(test - preds, 2))
print(mse)

# Plot the predictions with MSE
title = 'Predicting {0} via LTSM Network; MSE = {1}'.format(feat, mse)
train_plot = new_data[:17569]
test_plot = new_data[17569:]
test_plot['Predictions'] = preds
print(pd.to_datetime(train_plot.index))
print(pd.to_datetime(test_plot.index))
plt.plot_date(pd.to_datetime(train_plot.index), train_plot[feat], fmt='-')
plt.plot_date(pd.to_datetime(test_plot.index), test_plot[[feat, 'Predictions']], fmt='-')
plt.title(title)
plt.xlabel('Date')
plt.ylabel(feat)
plt.legend(['Training Data', 'Test Data', 'Test Prediction'])
plt.show()
