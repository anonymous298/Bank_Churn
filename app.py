import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
import streamlit as st

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        padding: 1rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 1rem;
    }
    /* Custom styling for success messages */
    .stSuccess {
        background-color: #D4EDDA;
        color: #155724;
        border-color: #C3E6CB;
    }
    /* Custom styling for error messages */
    .stError {
        background-color: #F8D7DA;
        color: #721C24;
        border-color: #F5C6CB;
    }
    .prediction-container-success {
        background-color: #D4EDDA;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .prediction-container-error {
        background-color: #F8D7DA;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Replace simple header with styled header
st.markdown('<h1 class="main-header">Bank Customer Churn Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Enter customer details to predict churn probability</p>', unsafe_allow_html=True)

def load_models():
    model = load_model('pickle_files/NN_model.h5')

    with open('pickle_files/labelencoder.pkl', 'rb') as file:
        le_gender = pickle.load(file)

    with open('pickle_files/ohe.pkl', 'rb') as file:
        ohe_geo = pickle.load(file)

    with open('pickle_files/scaling.pkl', 'rb') as file:
        scaler = pickle.load(file)

    return model, le_gender, ohe_geo, scaler

def class_returner(value):
    probability = value[0][0]
    if probability > 0.5:
        return "The person will likely to Churn", probability
    else:
        return 'The person will not likely to Churn', probability

def main():
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        creditscore = st.number_input('Credit Score 💳', 300, 850, 400, help="Customer's credit score (300-850)")
        geography = st.selectbox('Geography 🌍', ['France', 'Germany', 'Spain'])
        age = st.number_input('Age 👤', 10, 75, 25)
        
    with col2:
        gender = st.selectbox('Gender 👥', ['Male', 'Female'])
        balance = st.number_input('Balance 💰', 0, 10000000, 15000)
        tenure = st.number_input('Tenure 📅', 0, 10, 2, help="Years as a customer")
        
    with col3:
        numofproducts = st.selectbox('Number of Products 📦', [0,1,2,3,4], 2)
        hascrcard = st.selectbox('Has Credit Card 💳', [0, 1], 1, format_func=lambda x: 'Yes' if x == 1 else 'No')
        isactivemember = st.selectbox('Active Member Status ✅', [0,1], 1, format_func=lambda x: 'Active' if x == 1 else 'Inactive')
        
    estimatedsalary = st.slider('Estimated Salary 💵', 0, 1000000, 40000, format="$%d")

    input_querie = {
        'CreditScore' : creditscore, 
        'Geography' : geography, 
        'Gender' : gender, 
        'Age' : age, 
        'Tenure' : tenure, 
        'Balance' : balance,
        'NumOfProducts' : numofproducts, 
        'HasCrCard' : hascrcard, 
        'IsActiveMember' : isactivemember, 
        'EstimatedSalary' : estimatedsalary,
    }

    model, le_gender, ohe_geo, scaler = load_models()

    # Style the predict button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button('Predict Churn Probability', key='predict_button', help="Click to predict customer churn"):
        with st.spinner('Analyzing customer data...'):
            input_df = pd.DataFrame(input_querie, index=[0])

            input_df['Gender'] = le_gender.transform(input_df['Gender']) 

            ohe_col = ohe_geo.transform(input_df[['Geography']]).toarray()
            ohe_df = pd.DataFrame(ohe_col, columns=ohe_geo.get_feature_names_out())

            input_df = pd.concat([input_df.drop('Geography', axis=1), ohe_df], axis=1)

            scaled_input = scaler.transform(input_df)

            prediction = model.predict(scaled_input)

            final_prediction, probability = class_returner(prediction)
            probability_percentage = f"{probability:.1%}"

            # Style the prediction output
            if "likely to Churn" in final_prediction:
                st.error(f"⚠️ {final_prediction}")
                st.error(f"Churn Probability: {probability_percentage}")
            else:
                st.success(f"✅ {final_prediction}")
                st.success(f"Churn Probability: {probability_percentage}")

main()