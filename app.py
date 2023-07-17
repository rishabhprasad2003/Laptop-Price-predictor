import streamlit as st
import pickle
import numpy as np


df = pickle.load(open('df.pkl','rb'))
pipe = pickle.load(open('pipe.pkl','rb'))

st.header("LAPTOP PRICE PREDICTOR")

brand = st.selectbox("Brand Name",df['brand'].unique())

processor_brand = st.selectbox("Processor Brand",df['processor_brand'].unique())

processor_name = st.selectbox("Processor Name",df['processor_name'].unique())

processor_gnrtn = st.selectbox("Processor Generation",df['processor_gnrtn'].unique())

ram_gb = st.number_input("RAM (in GB)")

ram_type = st.selectbox("Ram Type",df['ram_type'].unique())

ssd = st.selectbox("SSD (in GB)",[0,8,16,32,64,128,256,512,1024])

hdd = st.selectbox("HDD (in GB)",[0,64,128,256,512,1024,2048])

os = st.selectbox("Operating system",df['os'].unique())

graphic_card_gb = st.number_input("Graphic card (in GB)")

weight_type = st.selectbox("Type of laptop",df['weight'].unique())

touch = st.selectbox("Touchscreen",["No","Yes"])

msoffice = st.selectbox("MS Office",["No","Yes"])

if (st.button("Predict Price")):
    if (touch=="Yes"):
        touch=1
    else:
        touch=0

    if (msoffice == "Yes"):
        msoffice=1
    else:
        msoffice=0

    inputs = np.array([ brand, processor_brand, processor_name, processor_gnrtn, ram_gb, ram_type, ssd, hdd, os, graphic_card_gb, weight_type, touch, msoffice],dtype=object)
    inputs = inputs.reshape(1,13)

    pred = pipe.predict(inputs)
    st.subheader("This is the predicted price: "+ str(np.exp(pred)[0]))
