import numpy as np
import pandas as pd 
import streamlit as st
import pickle

def load_model():
    with open('house_price_saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data['model']

def show_predict_page():
    st.title("House Price Prediction")

    st.write("""### We need some information to predict the Price""")

    # [CRIM	ZN	INDUS	CHAS	NOX	RM	AGE	DIS	RAD	TAX	PTRATIO	B	LSTAT]

    CRIM = st.text_input("CRIM:")
    ZN = st.text_input("ZN:")
    INDUS = st.text_input("INDUS:")
    CHAS = st.text_input("CHAS:")
    NOX = st.text_input("NOX:")
    RM = st.text_input("RM:")
    AGE = st.text_input("AGE:")
    DIS = st.text_input("DIS:")
    RAD = st.text_input("RAD:")
    TAX = st.text_input("TAX:")
    PTRATIO = st.text_input("PTRATIO:")
    B = st.text_input("B:")
    LSTAT = st.text_input("LSTAT:")

    ok = st.button("Predict")

    if ok:
        X = np.asarray([[CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT]])
        x_reshaped = X.reshape(1,-1)
        x_input = x_reshaped.astype(float)

        prediction = regressor.predict(x_input)
        st.success(f"The predicted price is: {prediction}" )
        