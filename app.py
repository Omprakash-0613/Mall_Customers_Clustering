import streamlit as st
import joblib
import numpy as np
model=joblib.load("model.pkl")
st.set_page_config(page_title="Mall Customer Segmentation",layout="centered")
st.title("Mall Customer Segmentation")
st.write("Enter the details of the customer to predict the segment.")

#income=st.slider("Annual Income (k$)",10,150,50)
#spending_score=st.slider("Spending Score (1-100)",1,100,50)

income=st.number_input("Annual Income (k$)",10,150,50)
spending_score=st.number_input("Spending Score (1-100)",1,100,50)
if st.button("Predict Segment"):
    input_data=np.array([[income,spending_score]])
    segment=model.predict(input_data)
    st.success(f"The customer belongs to Segment {segment[0]}")

