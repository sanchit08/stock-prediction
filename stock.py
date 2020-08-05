import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


df1 = pd.read_csv('/home/sanchit/Documents/machine learing/Datasets/NSE-TATAGLOBAL.csv')
df1['Date'] = pd.to_datetime(df1['Date'])
close = df1['Close'].to_list()
df = pd.DataFrame(index = range(0,len(df1)),columns = ['Date', 'Close'])
df['Date'] = df1['Date']
df['Close'] = close
df = df.set_index('Date')
dataset = df.values


l = len(df)
split = 0.9
features = 60 

train = dataset[0:int(l*split),:]
test = dataset[int(l*split):,:]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

x_train = []
y_train = []
x_test = []

for i in range(features,len(train)):
	x_train.append(scaled_data[i-features:i,0])
	y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


unit = 50
epoch = 15
batches = 32

model = Sequential()
model.add(LSTM(units = unit, return_sequences = True , input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = unit))
model.add(Dense(units = 1))


model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['accuracy'])

model.fit(x_train, y_train, epochs = epoch, batch_size = batches, verbose = 1)
model.save('stock.h5')
