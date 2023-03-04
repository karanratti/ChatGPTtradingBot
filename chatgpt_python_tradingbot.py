#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.optimizers import Adam

# Load the data
df = pd.read_csv('reliance_industries.csv')

# Split the data into training and testing sets
train_size = int(len(df) * 0.8)
train_data = df.iloc[:train_size, :].values
test_data = df.iloc[train_size:, :].values

# Define the input and output variables
X_train = train_data[:, 1:]
y_train = train_data[:, 0]
X_test = test_data[:, 1:]
y_test = test_data[:, 0]

# Train the GBM model
gbm = GradientBoostingRegressor()
gbm.fit(X_train, y_train)

# Evaluate the GBM model on the testing set
y_pred_gbm = gbm.predict(X_test)
rmse_gbm = np.sqrt(mean_squared_error(y_test, y_pred_gbm))
print('GBM RMSE:', rmse_gbm)

# Train the LSTM model
n_features = X_train.shape[1]
n_steps = 10
n_epochs = 50
batch_size = 32

X_train_lstm = np.zeros((X_train.shape[0] - n_steps + 1, n_steps, n_features))
y_train_lstm = y_train[n_steps - 1:]

for i in range(n_steps - 1, X_train.shape[0]):
    X_train_lstm[i - n_steps + 1] = X_train[i - n_steps + 1:i + 1]

model = Sequential()
model.add(LSTM(units=64, input_shape=(n_steps, n_features)))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

history = model.fit(X_train_lstm, y_train_lstm, epochs=n_epochs, batch_size=batch_size, verbose=0)

# Evaluate the LSTM model on the testing set
X_test_lstm = np.zeros((X_test.shape[0] - n_steps + 1, n_steps, n_features))
y_test_lstm = y_test[n_steps - 1:]

for i in range(n_steps - 1, X_test.shape[0]):
    X_test_lstm[i - n_steps + 1] = X_test[i - n_steps + 1:i + 1]

y_pred_lstm = model.predict(X_test_lstm)
rmse_lstm = np.sqrt(mean_squared_error(y_test_lstm, y_pred_lstm))
print('LSTM RMSE:', rmse_lstm)


# In[ ]:




