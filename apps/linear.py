from cProfile import label
import pandas as pd
import numpy as np
from sklearn import metrics 
import matplotlib.pyplot as plt
import pandas_datareader as data
import tensorflow as tf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, accuracy_score
import streamlit as st
import math
import datetime


def app():
    
    st.title('Linear Regression')
    #start = '2020-01-01'
    #end = '2021-12-31'
    start =  st.text_input('Enter Start Date','2020-01-01')
    end = st.text_input('Enter End Date','2021-12-31')
    st.title('Enter Your Crypto name Formate(BTC-USD)')
    user_input = st.text_input('Enter Crypto Ticker' ,'BTC-USD')

    
    dataset = data.DataReader('BTC-USD' ,'yahoo', start,end)
    
    dataset.shape
    dataset.drop('Adj Close',axis = 1, inplace = True)
    dataset.head()

    dataset.isnull().sum()
   
    st.subheader('Overall Performance')
    st.write(dataset.describe())
    st.write(len(dataset))
    

    

    st.subheader('Closing Price vs Time chart')
    fig = plt.figure(figsize= (16,6))
    plt.plot(dataset.Open)
    st.pyplot(fig)

    X  = dataset[['Open','High','Low','Volume']]
    y = dataset['Close']

    from sklearn.model_selection import train_test_split
    X_train , X_test , y_train , y_test = train_test_split(X ,y , random_state = 0)

    X_train.shape

    X_test.shape
    
    regressor = LinearRegression()
    regressor.fit(X_train,y_train)
    regressor.coef_
    regressor.intercept_
    predicted=regressor.predict(X_test)
    #print(X_test)
    predicted.shape
    dfr=pd.DataFrame(y_test,predicted)

    dfr=pd.DataFrame({'Actual':y_test,'Predicted':predicted})
    #print(dfr)
    #dfr.head(25)
    st.write(dfr.tail(10))

    st.write(regressor.score(X_test,y_test))

    graph=dfr.head(20)
    abk= graph.plot(kind='bar')
    #st.write(graph.plot(kind='bar'))
    st.write(abk)
    st.bar_chart(dfr.tail(20) )
    

   
    


    
