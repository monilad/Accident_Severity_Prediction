

import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import plotly.express as px


st.markdown("## 🚀 Accident Severity Prediction")
st.write("Please fill in the accident details below. The model will predict the expected **severity level**.")


# Load the model
model = joblib.load('best_xgb_model.pkl')
scaler = joblib.load('scaler.pkl')




# Input form
with st.form("input_form"):

    col1, col2 = st.columns(2)

with col1:
    light_condition = st.selectbox("Light Condition", ["Darklighted", "Daylight","Dusk","Dawn","Dark not lighted","Unknown"])
    weather = st.selectbox("Weather", ["Clear", "Cloudy","Rain","Unknown","Others","Fog"])
    surface_condition = st.selectbox("Surface Condition", ["Dry", "Unknown", "Wet","Others", "Stagnant Water"])
    weekday = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    rush_hour = st.selectbox("Rush Hour", ["Yes", "No"])

with col2:
    total_injuries = st.number_input("Total Injuries", min_value=0, max_value=100, value=0)
    total_fatalities = st.number_input("Total Fatalities", min_value=0, max_value=10, value=0)
    hour = st.slider("Hour of Accident", 0, 23, 12)
    month = st.selectbox("Month of Year", ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"])
    age_group = st.selectbox("Age Group", ["18-25", "26-60", "60+"]) 


    submitted = st.form_submit_button("Predict Severity")


    # light_condition = st.selectbox("Light Condition", ["Darklighted", "Daylight","Dusk","Dawn","Dark not lighted","Unknown"])
    # weather = st.selectbox("Weather", ["Clear", "Cloudy","Rain","Unknown","Others" "Fog"])
    # surface_condition = st.selectbox("Surface Condition", ["Dry", "Unknown", "Wet","Others", "Stagnant Water"])
    # total_injuries = st.number_input("Total Injuries", min_value=0, max_value=100, value=0)
    # total_fatalities = st.number_input("Total Fatalities", min_value=0, max_value=10, value=0)
    # hour = st.slider("Hour of Accident (0-23)", 0, 23, 12)
    # weekday = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
    # rush_hour = st.selectbox("Rush Hour", ["Yes", "No"])

    #submitted = st.form_submit_button("Predict Severity")


    # Preprocess inputs and predict
if submitted:
    # Example: Encode categorical values 
    light_map = {"Darklighted": 0, "Daylight": 1,"Dusk": 2,"Dawn": 3,"Dark not lighted": 4,"Unknown":5}
    weather_map = {"Clear": 0, "Cloudy":1 ,"Rain": 2,"Unknown": 3,"Others": 4, "Fog": 5}
    surface_map = {"Dry": 0, "Unknown": 1, "Wet": 2,"Others":3, "Stagnant Water": 4}
    weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    rush_map = {"No": 0, "Yes": 1}
    month_map = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
    age_group_map = {"18-25": 2, "26-60": 1, "60+": 0}




# Convert inputs to DataFrame (adjust feature order to match training data)
    # input_data = pd.DataFrame([[
    #     light_map[light_condition],
    #     weather_map[weather],
    #     surface_map[surface_condition],
    #     total_injuries,
    #     total_fatalities,
    #     hour,
    #     weekday_map[weekday],
    #     rush_map[rush_hour]
    # ]], columns=["Lightcondition", "Weather", "SurfaceCondition", "Totalinjuries", "Totalfatalities", "Hour", "Weekday", "Rush_Hour"])


    input_dict = {
    'Year': 2025,
    'Distance': 0,
    'Totalinjuries': total_injuries,
    'Totalfatalities': total_fatalities,
    'Collisionmanner': 6,
    'Lightcondition': light_map[light_condition],
    'Weather': weather_map[weather],
    'SurfaceCondition': surface_map[surface_condition],
    'Unittype_One': 0,
    'Gender_Drv1': 1,
    'Traveldirection_One': 1,
    'AlcoholUse_Drv1': 1,
    'DrugUse_Drv1': 1,
    'Unittype_Two': 0,
    'Gender_Drv2': 1,
    'Traveldirection_Two': 1,
    'AlcoholUse_Drv2': 1,
    'DrugUse_Drv2': 1,
    'Hour': hour,
    'Weekday': weekday_map[weekday],
    'Month': month_map[month],
    'Weekend': 0,
    'Rush_Hour': rush_map[rush_hour],
    'Hazardous_Road': 0,
    'Age_Group_Drv1': age_group_map[age_group],
    'Age_Group_Drv2': 1,
    'Substance_Use': 0,
    'Junction_Category': 4,
    'Violation_Category_Drv1': 2,
    'Violation_Category_Drv2': 5,
    'Unitaction_Category_Two': 3,
    'Unitaction_Category_One': 3
}

    input_df = pd.DataFrame([input_dict])
    scaled_input = scaler.transform(input_df)  # input_df must match original feature order
    #prediction = model.predict(scaled_input)



    # Predict
    prediction = model.predict(scaled_input)[0]
    severity_map = {0: "Fatal", 1: "Major Injury", 2: "Minor Injury", 3: "No Injury"}
    st.success(f"Predicted Severity: **{severity_map[prediction]}**")
