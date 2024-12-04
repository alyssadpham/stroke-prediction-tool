#Stroke Prediction Model - Application

#add libraries
import streamlit as st
import pandas as pd
import joblib

#load the pre-trained model and feature names
model=joblib.load('stroke_prediction_model.pkl')
feature_names=joblib.load('model_feature_names.pkl')

#add app title
st.title("Stroke Prediction Application")

# Collect user inputs
st.header("Patient Data Input")
gender=st.selectbox("Biological Sex",["Male","Female"])
age=st.number_input("Age (in years)",min_value=0,max_value=120,value=30)
hypertension=st.selectbox("Hypertension History",["No","Yes"])
heart_disease=st.selectbox("Heart Disease History",["No", "Yes"])
ever_married=st.selectbox("Marital Status",["No", "Yes"])
work_type = st.selectbox(
    "Employment Type",
    ["Private", "Self-employed", "Government Job", "Children", "Unemployed"]
)
residence_type=st.selectbox("Residence Type",["Urban", "Rural"])
avg_glucose_level=st.slider("Average Glucose Level (mg/dL)",min_value=50.0,max_value=300.0,step=0.1,value=100.0)
bmi=st.slider("Body Mass Index (BMI)",min_value=10.0,max_value=50.0,step=0.1,value=25.0)
smoking_status=st.selectbox("Smoking Status",["Never Smoked","Formerly Smoked","Currently Smokes"])

# Prepare user input data
input_data=pd.DataFrame({
    'gender_Male':[1 if gender == "Male" else 0],
    'age':[age],
    'hypertension':[1 if hypertension == "Yes" else 0],
    'heart_disease':[1 if heart_disease == "Yes" else 0],
    'ever_married_Yes':[1 if ever_married == "Yes" else 0],
    'work_type_Private':[1 if work_type == "Private" else 0],
    'work_type_Self-employed':[1 if work_type == "Self-employed" else 0],
    'work_type_Children':[1 if work_type == "Children" else 0],
    'work_type_Unemployed':[1 if work_type == "Unemployed" else 0],
    'Residence_type_Urban':[1 if residence_type == "Urban" else 0],
    'avg_glucose_level':[avg_glucose_level],
    'bmi':[bmi],
    'smoking_status_Formerly Smoked':[1 if smoking_status == "Formerly Smoked" else 0],
    'smoking_status_Currently Smokes':[1 if smoking_status == "Currently Smokes" else 0]
})

#align input data to match model's expected features
input_data_aligned=pd.DataFrame(columns=feature_names)
input_data_aligned.loc[0]=0 #initialize all features with zeros
input_data_aligned.update(input_data)  #update them with user inputs

#prediction action
if st.button("Predict Stroke Risk"):
    prediction=model.predict(input_data_aligned)[0]
    if prediction == 1:
        st.error("High Risk: The model predicts a high risk of stroke. Consult a healthcare professional.")
    else:
        st.success("Low Risk: The model predicts a low risk of stroke.")

#add customized footer
st.write("**Disclaimer:** This tool is for educational purposes only and should not replace professional medical advice.")