# Gender Male -> 1, Female -> 0
# Subscription Type Basic -> 1, Premium -> 2, Standard -> 0   
# Contract Length Monthly -> 1, Yearly -> 2, Quarterly -> 0
# Train-test split with random_state=42 for reproducibility
# Scaler imported as Scaler.pkl
# Model imported as best_model.pkl
# Order of inputs: 'Age', 'Gender', 'Tenure', 'Payment Delay', 'Subscription Type', 'Contract Length'

import sklearn
import streamlit as st
import numpy as np
import joblib

model = joblib.load('best_model.pkl')

scaler = joblib.load('scaler.pkl')


st.title('Customer Churn Prediction')
st.divider()

st.write("Please input the following customer details to predict churn:")

st.divider()
age = st.number_input('Age', min_value=18, max_value=100, value=30)
gender = st.selectbox('Gender', options=['Male', 'Female'])
tenure = st.number_input('Tenure (months)', min_value=0, max_value=120, value=12)
payment_delay = st.number_input('Payment Delay (days)', min_value=0, max_value=365, value=0)
subscription_type = st.selectbox('Subscription Type', options=['Basic', 'Premium', 'Standard'])
contract_length = st.selectbox('Contract Length', options=['Monthly', 'Yearly', 'Quarterly'])
st.divider()  

predictbutton = st.button('Predict Churn', key='predict_button')

if predictbutton:
    gender_encoded = 1 if gender == 'Male' else 0
    subscription_encoded = 1 if subscription_type == 'Basic' else (2 if subscription_type == 'Premium' else 0)
    contract_encoded = 1 if contract_length == 'Monthly' else (2 if contract_length == 'Yearly' else 0)             
    input_data = np.array([[age, gender_encoded, tenure, payment_delay, subscription_encoded, contract_encoded]])           
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)   
    if prediction == 1:
        st.error('The customer is likely to churn.', icon="🚨")
    else:
        st.balloons()
        st.success('The customer is unlikely to churn.', icon="✅")
else:
    st.write('Click the "Predict Churn" button to see the result.')
