import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

df = pd.read_csv('/home/sanchit/Documents/machine learing/Datasets/NSE-TATAGLOBAL.csv')
df['Date'] = pd.to_datetime(df['Date'])
dff = pd.DataFrame(index = range(0,len(df)),columns = ['Date', 'Close'])
close = df['Close'].to_list()
dff['Date'] = df['Date']
dff['Close'] = close
dff = dff.set_index('Date')
n = int(len(dff)*0.2)
dff = dff[-n:]
dates = df['Date'].to_list()
dataset = dff.values
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

x_test = []
y_test = []

for i in range(60,len(dataset)):
	x_test.append(scaled_data[i-60:i,0])
	y_test.append(scaled_data[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


model = load_model('stock.h5')
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

pred = model.predict(x_test)
pred = np.reshape(pred,(1,-1))
pred = scaler.inverse_transform(pred)
pred = pred.ravel()


y_test = np.reshape(y_test,(1,-1))
y_test = scaler.inverse_transform(y_test)
y_test = y_test.ravel()

plt.figure()
plt.plot(pred)
plt.plot(y_test)
plt.show()
