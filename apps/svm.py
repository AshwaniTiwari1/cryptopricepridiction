from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import tensorflow as tf
import seaborn as sns
from keras.models import load_model
import streamlit as st


def app():
    st.title('SVM Model')

    #start = '2010-01-01'
    #end = '2019-12-31'

    start = st.text_input('Enter Start Date', '2020-01-01')
    end = st.text_input('Enter End Date', '2021-12-31')

    st.title('Enter Your Crypto name Formate(BTC-USD)')

    user_input = st.text_input('Enter Crypto Ticker', 'BTC-USD')

    df = data.DataReader(user_input, 'yahoo', start, end)
   #df = data.DataReader('BTC-USD' ,'yahoo', start,end)

    # Describing DATA

    st.subheader('Overall Performance')
    st.write(df.describe())

    st.subheader("""User Input data Show""")
    st.write(df.head(5))
    st.write(df.tail(5))

    st.write(df.isnull().sum())

    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize=(12, 6))
    plt.plot(df.Close)
    st.pyplot(fig)

    df.drop(['Open', 'High', 'Low', 'Adj Close', 'Volume'],
            1, inplace=True)  # drop column date

    st.write(df.head())
    pred = 30 #n = 30
    df['Prediction'] = df[['Close']].shift(-pred) #adding the pred value to the dataframe
    st.write(df.head())

    #st.write(df.tail())

    x = np.array(df.drop(['Prediction'],1)) #Drop the prediction column and convert the dataframe into array 
    x = x[:len(df)-pred] #Removing last 'n' rows
   # print(x)

    y = np.array(df['Prediction']) #Drop the prediction column and convert the dataframe into array 
    y = y[:-pred] #Removing last 'n' rows
    #print(y)

    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x,y, test_size = 0.2)

    pred_arr = np.array(df.drop(['Prediction'],1))[-pred:]
    print(pred_arr)


    from sklearn.svm import SVR
    s = SVR(kernel = 'rbf', C = 1e3, gamma = 0.0001)
    s.fit(xtrain,ytrain)
    #Creating and training the Support Vector Machine using radial basis function
    s_c = s.score(xtest,ytest)*-100

    s_p = s.predict(xtest)
    st.write(s_p)

    print(ytest)

    print(df.tail(pred))

    svm_p = s.predict(pred_arr)
    print(svm_p)

    df1 = pd.DataFrame(svm_p, columns =['Predicted Price']) #converting the array into dataframe
    df1 

    df['Prediction'] = df1 #adding the predicted price to the original dataframe
    df.head()
    st.write(df.head())

