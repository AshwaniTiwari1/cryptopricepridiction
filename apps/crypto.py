import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets

def app():
    st.markdown('''# **Cryptocurrency Price App**
A simple cryptocurrency price app pulling price data.
''')

st.header('**Selected Price**')

# Load market data from Binance API
df = pd.read_json('https://api.binance.com/api/v3/ticker/24hr')


st.header('**All Price**')
st.dataframe(df)

st.info('')