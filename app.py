import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import load_model
import numpy as np

from tensorflow import keras

st.title('Commercial bank Stock Price prediction')
st.write('DSCS University of kelaniya')
with st.form("my_form"):
    
    number =  st.number_input('Insert  Low trade price', key ='0' )
    number1 = st.number_input('Insert  close price', key  = '1')
    number2 = st.number_input('Insert  Opening price', key = '2')
    number3 = st.number_input('Insert  No of Trades', key = '3')
    number4 = st.number_input('Insert  Volume', key = '4')
    number5 = st.number_input('Insert  Trunover', key = '5')
    list_enterd = [number,number1,number2,number3,number4,number5]
    custom_df = pd.DataFrame(list_enterd).T
    custom_df.columns = ['Low', 'Close', 'Open', 'Trades', 'Volume', 'Trunover']
    cols =  list(custom_df)
    custom_df=custom_df[cols].astype(float)
    custom_df_array = np.array(custom_df)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_custom_scaled = scaler.fit_transform(custom_df_array)
    custom_scaler_list = np.repeat(df_custom_scaled, 14,axis=0)
    custom_scaler_list = np.repeat(custom_scaler_list[np.newaxis,:,:],882, axis=0)
    custom_scaler_list = np.array(custom_scaler_list)
    modelnew = load_model('C:\Users\sachin.dilhan\Desktop\COMB\stock_comb\150 epoch-20220108T180709Z-001')
    list_Custom_prediction = modelnew.predict(custom_scaler_list)
    list_forcast_copies_finalize = np.repeat(list_Custom_prediction, 6, axis=-1)
    list_predictions_333 = scaler.inverse_transform(list_forcast_copies_finalize)[:,0]
    format_float = "{:.2f}".format(list_predictions_333[0])
    # Every form must have a submit button.
    submitted = st.form_submit_button("Submit")
if submitted:
        
    
  st.title("Maximum trading  prediction price is ",format_float)
  html_str = f"""
  <style>
  p.a {{
   font: bold 35px Courier;
  }}
  </style>
  <p class="a">Rs {format_float}/=</p>
  """
  st.markdown(html_str, unsafe_allow_html=True)