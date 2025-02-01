import numpy as np
import streamlit as st
import pickle
import pandas as pd

with open('Passenger.pkl','rb') as file:
    loaded_model=pickle.load(file)

st.title('Passenger Survival')
st.subheader('Survival')

df=pd.read_csv('survival.csv')

column_list=df.columns.to_list()

uploaded_file=st.file_uploader('Upload your csv file',type=['csv'])

if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)
    df=df.reindex(columns=column_list,fill_value=0)
    prediction = loaded_model.predict(df)
    prediction_text = np.where(prediction == 1, 'Yes', 'No')
    st.subheader('Survived:')
    st.write(prediction_text)