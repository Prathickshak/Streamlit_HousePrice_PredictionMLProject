import streamlit as st
from predict_page import show_predict_page

page = st.sidebar.selectbox("Welcome to House Price Prediction", ("Get Started", "Nothanks"))

if page == "Get Started":
    show_predict_page()
else:
    qoutes = "Thanks for visiting our page..."
    st.success(qoutes)