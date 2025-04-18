import streamlit as st
import pandas as pd
import numpy as np
import sklearn
import pickle

model = pickle.load(open('linear.pkl','rb')) #read binary

# let's create the web app:
st.title("Scikit Learn Linear Regresssion Model")

tv = st.text_input("Enter TV sales: ")
radio = st.text_input("Enter radio sales: ")
newspaper = st.text_input("Enter newspaper sales: ")
if st.button("predict"):
    features = np.array([[tv,radio,newspaper]],dtype = np.float64)
    results = model.predict(features).reshape(1,-1)
    st.write("Predicted Sale::::: ",results[0])

# streamlit run lr_app.py =>command to run the file open the terminal and enter the command