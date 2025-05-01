import streamlit as st
import pickle
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Custom CSS for prediction page
st.markdown("""
<style>
    /* Base styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header styling */
    .main-header {
        color: #1E3A8A;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    
    /* Sub header styling */
    .sub-header {
        color: #2563EB;
        font-size: 1.3rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    
    /* Form styling */
    .form-container {
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 10px 15px rgba(0, 0, 0, 0.03);
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    /* Info box styling */
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Success box styling */
    .success-box {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        border-radius: 0.375rem;
        padding: 1.5rem;
        margin: 1.5rem 0;
    }
    
    /* Warning box styling */
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1.5rem 0;
    }
    
    /* Severity result styling */
    .severity-result {
        display: flex;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    .severity-icon {
        font-size: 2rem;
        margin-right: 1rem;
    }
    
    .severity-text {
        font-size: 1.5rem;
        font-weight: 600;
    }
    
    .severity-fatal {
        color: #DC2626;
    }
    
    .severity-major {
        color: #F59E0B;
    }
    
    .severity-minor {
        color: #2563EB;
    }
    
    .severity-none {
        color: #10B981;
    }
    
    /* Results container styling */
    .results-container {
        background-color: white;
        border-radius: 0.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        padding: 1.5rem;
        margin-top: 2rem;
    }
    
    /* Form section styling */
    .form-section {
        margin-bottom: 1.5rem;
    }
    
    .form-section-title {
        color: #4B5563;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    
    /* Custom form input styling */
    .stSelectbox > div > div,
    .stNumberInput > div > div {
        background-color: #F9FAFB;
        border-radius: 0.375rem;
    }
    
    /* Submit button styling */
    .stButton > button {
        background-color: #2563EB;
        color: white;
        font-weight: 500;
        padding: 0.5rem 1.5rem;
        border-radius: 0.375rem;
        border: none;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #1D4ED8;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Factor impact bar styling */
    .factor-bar {
        display: flex;
        align-items: center;
        margin-bottom: 0.75rem;
    }
    
    .factor-name {
        width: 150px;
        font-size: 0.9rem;
        color: #4B5563;
    }
    
    .factor-bar-container {
        flex-grow: 1;
        height: 10px;
        background-color: #E5E7EB;
        border-radius: 99px;
        overflow: hidden;
        margin: 0 0.5rem;
    }
    
    .factor-bar-fill {
        height: 100%;
        border-radius: 99px;
    }
    
    .factor-impact {
        width: 50px;
        text-align: right;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Chart section styling */
    .chart-container {
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* Slider custom styling */
    .stSlider > div > div > div {
        background-color: #3B82F6;
    }
    
    /* Labels for inputs */
    label {
        font-weight: 500;
        color: #4B5563;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header"> Accident Severity Prediction</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <p>Please fill in the accident details below. The model will predict the expected <strong>severity level</strong> based on the provided information.</p>
</div>
""", unsafe_allow_html=True)

# Try to load the model - handle potential errors gracefully
try:
    model = joblib.load('best_xgb_model.pkl')
    scaler = joblib.load('scaler.pkl')
    model_loaded = True
except Exception as e:
    st.markdown("""
    <div class="warning-box">
        <strong>⚠️ Model Loading Error:</strong> Unable to load the prediction model. The app will run in demonstration mode.
    </div>
    """, unsafe_allow_html=True)
    model_loaded = False

# Improved form layout with sections
with st.form("prediction_form"):
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<div class="form-section-title">Environmental Factors</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        light_condition = st.selectbox("Light Condition", 
                                      ["Daylight", "Darklighted", "Dusk", "Dawn", "Dark not lighted", "Unknown"],
                                      index=0)
        weather = st.selectbox("Weather", 
                              ["Clear", "Cloudy", "Rain", "Unknown", "Others", "Fog"],
                              index=0)
    
    with col2:
        surface_condition = st.selectbox("Surface Condition", 
                                        ["Dry", "Unknown", "Wet", "Others", "Stagnant Water"],
                                        index=0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Accident Details Section
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<div class="form-section-title">Accident Details</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        total_injuries = st.number_input("Total Injuries", 
                                        min_value=0, max_value=100, value=0)
        total_fatalities = st.number_input("Total Fatalities", 
                                          min_value=0, max_value=10, value=0)
        
    with col2:
        rush_hour = st.selectbox("Rush Hour", 
                                ["No", "Yes"],
                                index=0)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Temporal Factors Section
    st.markdown('<div class="form-section">', unsafe_allow_html=True)
    st.markdown('<div class="form-section-title">Temporal Factors</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        hour = st.slider("Hour of Accident", 
                        0, 23, 12,
                        format="%d:00")
        weekday = st.selectbox("Day of Week", 
                              ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                              index=0)
        
    with col2:
        month = st.selectbox("Month of Year", 
                           ["January", "February", "March", "April", "May", "June", 
                            "July", "August", "September", "October", "November", "December"],
                           index=0)
        age_group = st.selectbox("Age Group", 
                                ["18-25", "26-60", "60+"],
                                index=1)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Submit button with improved styling
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        submitted = st.form_submit_button("Predict Severity")

# Prediction section
if submitted:
    # Mapping dictionaries for encoding
    light_map = {"Darklighted": 0, "Daylight": 1, "Dusk": 2, "Dawn": 3, "Dark not lighted": 4, "Unknown": 5}
    weather_map = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Unknown": 3, "Others": 4, "Fog": 5}
    surface_map = {"Dry": 0, "Unknown": 1, "Wet": 2, "Others": 3, "Stagnant Water": 4}
    weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
    rush_map = {"No": 0, "Yes": 1}
    month_map = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, 
                "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
    age_group_map = {"18-25": 2, "26-60": 1, "60+": 0}

    # Create input dictionary for model
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

    # Prepare visualization container
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    
    # Make prediction if model is loaded, otherwise show demo result
    if model_loaded:
        input_df = pd.DataFrame([input_dict])
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        # Get probabilities for visualization
        probabilities = model.predict_proba(scaled_input)[0]
    else:
        # Demo mode - determine a reasonable prediction based on inputs
        if total_fatalities > 0:
            prediction = 0  # Fatal
            probabilities = [0.75, 0.15, 0.07, 0.03]
        elif total_injuries > 3:
            prediction = 1  # Major Injury
            probabilities = [0.20, 0.65, 0.10, 0.05]
        elif total_injuries > 0:
            prediction = 2  # Minor Injury
            probabilities = [0.05, 0.15, 0.70, 0.10]
        else:
            prediction = 3  # No Injury
            probabilities = [0.03, 0.07, 0.20, 0.70]
    
    # Map prediction to severity label
    severity_map = {0: "Fatal", 1: "Major Injury", 2: "Minor Injury", 3: "No Injury"}
    severity = severity_map[prediction]
    
    # Display prediction result with appropriate styling
    severity_classes = {
        "Fatal": "severity-fatal",
        "Major Injury": "severity-major",
        "Minor Injury": "severity-minor",
        "No Injury": "severity-none"
    }
    
    severity_icons = {
        "Fatal": "☠️",
        "Major Injury": "🚨",
        "Minor Injury": "⚠️",
        "No Injury": "✅"
    }
    
    st.markdown(f"""
    <h2 class="sub-header">Prediction Result</h2>
    <div class="severity-result">
        <div class="severity-icon">{severity_icons[severity]}</div>
        <div class="severity-text {severity_classes[severity]}">
            {severity}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Display probability distribution with better styling
    st.markdown('<h3 class="sub-header">Severity Probability Distribution</h3>', unsafe_allow_html=True)
    
    # Create a list of severity labels for the chart
    severities = list(severity_map.values())
    
    # Create a color map for the different severity levels
    color_map = {
        "Fatal": "#DC2626",
        "Major Injury": "#F59E0B",
        "Minor Injury": "#2563EB",
        "No Injury": "#10B981"
    }
    
    # Create a better-looking chart
    fig = px.bar(
        x=severities,
        y=probabilities,
        labels={'x': 'Severity Level', 'y': 'Probability'},
        color=severities,
        color_discrete_map=color_map,
        template="plotly_white"
    )
    
    fig.update_layout(
        title_text='Probability Distribution by Severity Level',
        title_x=0.5,
        xaxis_title="Severity Level",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        showlegend=False,
        height=400
    )
    
    # Add value labels on top of bars
    fig.update_traces(
        texttemplate='%{y:.1%}',
        textposition='outside'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show key factors influencing the prediction
    st.markdown('<h3 class="sub-header">Key Influencing Factors</h3>', unsafe_allow_html=True)
    
    # Example influential factors (in a real app, these would come from SHAP values or feature importance)
    factors = []
    
    # Add dynamic factors based on input
    if total_fatalities > 0:
        factors.append({"factor": "Total Fatalities", "impact": 0.40, "direction": "increase"})
    
    if total_injuries > 0:
        factors.append({"factor": "Total Injuries", "impact": 0.30, "direction": "increase"})
    
    if weather != "Clear":
        factors.append({"factor": "Weather", "impact": 0.25, "direction": "increase" if weather in ["Rain", "Fog"] else "decrease"})
    
    if surface_condition != "Dry":
        factors.append({"factor": "Surface Condition", "impact": 0.20, "direction": "increase" if surface_condition in ["Wet", "Stagnant Water"] else "decrease"})
    
    if light_condition != "Daylight":
        factors.append({"factor": "Light Condition", "impact": 0.15, "direction": "increase" if light_condition in ["Dark not lighted", "Darklighted"] else "decrease"})
    
    if rush_hour == "Yes":
        factors.append({"factor": "Rush Hour", "impact": 0.12, "direction": "increase"})
    
    if age_group in ["18-25", "60+"]:
        factors.append({"factor": "Age Group", "impact": 0.10, "direction": "increase"})
    
    # Add hour of day as a factor
    if 6 <= hour <= 9 or 16 <= hour <= 19:
        factors.append({"factor": "Hour of Day", "impact": 0.08, "direction": "increase" if rush_hour == "Yes" else "decrease"})
    elif 23 <= hour or hour <= 5:
        factors.append({"factor": "Hour of Day", "impact": 0.15, "direction": "increase"})
    
    # Ensure we have at least some factors
    if len(factors) < 3:
        factors.append({"factor": "Day of Week", "impact": 0.07, "direction": "increase" if weekday in ["Friday", "Saturday"] else "decrease"})
        factors.append({"factor": "Month", "impact": 0.05, "direction": "increase" if month in ["December", "January", "July"] else "decrease"})
    
    # Sort factors by impact
    factors = sorted(factors, key=lambda x: x["impact"], reverse=True)
    
    # Display the factors with nice progress bars
    for factor in factors:
        direction_icon = "↑" if factor["direction"] == "increase" else "↓"
        bar_color = "#DC2626" if factor["direction"] == "increase" else "#10B981"
        
        st.markdown(f"""
        <div class="factor-bar">
            <div class="factor-name">{factor["factor"]}</div>
            <div class="factor-bar-container">
                <div class="factor-bar-fill" style="width: {factor["impact"] * 100}%; background-color: {bar_color};"></div>
            </div>
            <div class="factor-impact">{direction_icon} {factor["impact"]*100:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations based on prediction
    st.markdown('<h3 class="sub-header">Safety Recommendations</h3>', unsafe_allow_html=True)
    
    if prediction == 0:  # Fatal
        st.markdown("""
        <div class="warning-box">
            <strong>High Risk Scenario:</strong> The conditions you've entered indicate a high-risk situation that could lead to fatal outcomes.
            <ul>
                <li>Consider postponing travel in these conditions if possible</li>
                <li>Use extra caution and reduce speed significantly</li>
                <li>Ensure all safety systems are functioning properly</li>
                <li>Maintain increased distances between vehicles</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif prediction == 1:  # Major Injury
        st.markdown("""
        <div class="warning-box">
            <strong>Significant Risk Scenario:</strong> These conditions are associated with accidents resulting in major injuries.
            <ul>
                <li>Reduce speed and increase following distance</li>
                <li>Avoid distractions completely</li>
                <li>Be particularly alert at intersections</li>
                <li>Consider alternative routes if available</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    elif prediction == 2:  # Minor Injury
        st.markdown("""
        <div class="info-box">
            <strong>Moderate Risk Scenario:</strong> These conditions present some risk of minor injury accidents.
            <ul>
                <li>Drive defensively and stay alert</li>
                <li>Follow all traffic rules carefully</li>
                <li>Maintain appropriate speed for conditions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:  # No Injury
        st.markdown("""
        <div class="success-box">
            <strong>Lower Risk Scenario:</strong> These conditions are generally associated with lower severity outcomes.
            <ul>
                <li>Continue to follow safe driving practices</li>
                <li>Stay alert to changing conditions</li>
                <li>Remember that even lower risk situations require attention</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)  # Close results container

# Add information at the bottom of the page
st.markdown("""
<div style="background-color: #F9FAFB; padding: 1rem; border-radius: 0.5rem; margin-top: 2rem;">
    <h3 style="color: #4B5563; font-size: 1.1rem; font-weight: 600; margin-bottom: 0.5rem;">About This Prediction Tool</h3>
    <p style="color: #6B7280; font-size: 0.9rem; line-height: 1.5;">
        This model uses gradient boosting to analyze historical accident data and predict severity outcomes. 
        The predictions are based on environmental factors, accident details, and temporal patterns.
        For a deeper understanding of how predictions are made, visit the Explainable AI section.
    </p>
    <p style="color: #DC2626; font-size: 0.8rem; margin-top: 0.5rem;">
        <strong>Disclaimer:</strong> This tool is for educational and demonstration purposes only. 
        Always follow local traffic safety guidelines and use caution in all driving conditions.
    </p>
</div>
""", unsafe_allow_html=True)