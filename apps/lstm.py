from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import tensorflow as tf
from keras.models import load_model
import streamlit as st

def app():
    st.title('LSTM Model')

    
     
    #start = '2010-01-01'
    #end = '2019-12-31'

    start =  st.text_input('Enter Start Date','2020-01-01')
    end = st.text_input('Enter End Date','2021-12-31')


    st.title('Enter Your Crypto name Formate(BTC-USD)')

    user_input = st.text_input('Enter Crypto Ticker' ,'BTC-USD')

    df = data.DataReader(user_input ,'yahoo', start,end)
   #df = data.DataReader('BTC-USD' ,'yahoo', start,end)

    #Describing DATA

    st.subheader('Overall Performance')
    st.write(df.describe())
    
    st.subheader("""User Input data Show""")
    st.write(df.head(5))
    st.write(df.tail(5))

    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize= (12,6))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA ')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize= (12,6))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time chart with 100MA & 200MA')
    ma100 = df.Close.rolling(100).mean()
    ma200 = df.Close.rolling(200).mean()
    fig = plt.figure(figsize= (12,6))
    plt.plot(ma100)
    plt.plot(ma200)
    plt.plot(df.Close)
    st.pyplot(fig)

   # Spliting DATA into Training and Testing

    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

    print(data_training.shape)
    print(data_testing.shape)


    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)



    model = load_model('keras_model.h5')

    past_100_days = data_training.tail(100)

    final_df = past_100_days.append(data_testing, ignore_index=True)

    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
       x_test.append(input_data[i-100: i])
       y_test.append(input_data[i, 0])


    x_test, y_test = np.array(x_test), np.array(y_test)

    y_predicted = model.predict(x_test)

    scaler = scaler.scale_

    scaler_factor = 1/scaler[0]
    y_predicted = y_predicted * scaler_factor
    y_test = y_test * scaler_factor
    st.header('prediction value')
    st.write(y_predicted )
   # st.write(y_test)
    #final Graph
    st.subheader('Prediction Vs Orginal')
    fig2 = plt.figure(figsize= (12,6))
    plt.plot(y_test, 'b', label = 'Orginal Price')
    plt.plot(y_predicted, 'g', label = 'Prediction Price')
    plt.xlabel('Time')
    plt.ylabel('price')
    plt.legend()
    st.pyplot(fig2)
    plt.show()
