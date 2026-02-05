import pandas as pd 
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle
import streamlit as st


#Without Keras, writing ANN code is long and complex.Keras automatically manages:Forward propagation,Backpropagation,Weight updates,Gradient calculation,Optimization


#load the trained models
model=tf.keras.models.load_model('model.h5')


##load the encoder and scalar 
with open('onehot_encoder_geo.pkl','rb') as file:  #rb-read binary
    onehot_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)


#streamlit app
st.title('Customer Churn Predection')

# for getting the detail of user input how its working go on-->"https://chatgpt.com/c/6984d321-1dfc-8322-b43e-426052e75b95"
# User input(just like a form)
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({  ##Pandas expects columns as lists that's why we are using xyz=[xyz]
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]], # ML model cannot understand text, Gender was Label Encoded during training
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})



# One-hot encode for 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine one-hot encoded columns with input data   -->(Germany,spain and france will be added after this concatenation with input_data in original data)
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data  -->->data will converted into array
input_data_scaled = scaler.transform(input_data)


#for predection
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Churn Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to churn.')
else:
    st.write('The customer is not likely to churn.')

