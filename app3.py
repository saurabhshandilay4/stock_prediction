import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
from datetime import date
import math


start = '2010-01-01'
end = date.today().strftime("%Y-%m-%d")


st.title('Stock Prediction')
user_input=st.text_input('Enter Stock Ticker', 'AAPL')

df = data.DataReader(user_input, 'yahoo', start, end)


#describing data
st.subheader('Data from 2010-Present')
st.write(df.describe())

#visualizations
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

df1=df.reset_index()['Close']

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


training_size=int(len(df1)*0.65)
test_size=len(df1)-training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]


def create_dataset(dataset, time_step=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


time_step = 100
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)


X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


model = load_model(r'./keras_model65.h5', custom_objects=None, compile=True)

train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


look_back=100
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
st.subheader('Prediction vs Original')
fig2=plt.figure(figsize=(12, 6))
plt.plot(scaler.inverse_transform(df1),'b', label='Original Price')
plt.plot(trainPredictPlot,'y', label='Train predicted Price')
plt.plot(testPredictPlot,'r', label='Test Original Price')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


x=len(test_data)-100

x_input=test_data[x:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()


from numpy import array

lst_output=[]
n_steps=100
i=0
while(i<30):
    
    if(len(temp_input)>100):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    


day_new=np.arange(1,101)
day_pred=np.arange(101,131)


y=len(df1)-100

st.subheader('Forcasting for next 30 days')
fig3 = plt.figure(figsize=(12, 6))
plt.plot(day_new,scaler.inverse_transform(df1[y:]))
plt.plot(day_pred,scaler.inverse_transform(lst_output))
st.pyplot(fig3)


df3=df1.tolist()
df3.extend(lst_output)

st.subheader('Extended')
fig4 = plt.figure(figsize=(12, 6))
plt.plot(df3[y:])
st.pyplot(fig4)

df3=scaler.inverse_transform(df3).tolist()

st.subheader('Final Result')
fig5 = plt.figure(figsize=(12, 6))
plt.plot(df3)
st.pyplot(fig5)
