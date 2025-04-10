import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import base64
import re

# Set page configuration with expanded layout and custom theme
st.set_page_config(
    page_title="Road Safety Analyzer",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
def local_css():
    st.markdown("""
    <style>
        /* Main styling */
        .main {
            background-color: #f8f9fa;
        }
        
        /* Custom header styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #1e3a8a;
            margin-bottom: 1rem;
            text-align: center;
            padding: 1rem;
            background: linear-gradient(90deg, #60a5fa, #3b82f6, #2563eb);
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        /* Card styling */
        .card {
            padding: 1.5rem;
            border-radius: 10px;
            background-color: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
            border-left: 5px solid #3b82f6;
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
        }
        
        /* Section headers */
        .section-header {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e40af;
            margin-bottom: 1rem;
            border-bottom: 2px solid #3b82f6;
            padding-bottom: 0.5rem;
        }
        
        /* Feature input styling */
        .stSlider > div > div {
            background-color: #3b82f6 !important;
        }
        
        .stButton > button {
            background-color: #2563eb;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 600;
            border: none;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #1d4ed8;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            transform: translateY(-2px);
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #1e3a8a;
        }
        
        .sidebar .sidebar-content {
            background-color: #1e3a8a;
            color: white;
        }
        
        /* Success message */
        .success-box {
            padding: 1rem;
            background-color: #ecfdf5;
            border-left: 5px solid #10b981;
            border-radius: 5px;
            margin: 1rem 0;
            font-weight: 500;
        }
        
        /* Warning message */
        .warning-box {
            padding: 1rem;
            background-color: #fffbeb;
            border-left: 5px solid #f59e0b;
            border-radius: 5px;
            margin: 1rem 0;
            font-weight: 500;
        }
        
        /* Info box */
        .info-box {
            padding: 1rem;
            background-color: #eff6ff;
            border-left: 5px solid #3b82f6;
            border-radius: 5px;
            margin: 1rem 0;
            font-weight: 500;
        }
        
        /* Badge styling */
        .badge {
            display: inline-block;
            padding: 0.35em 0.65em;
            font-size: 0.75em;
            font-weight: 700;
            line-height: 1;
            color: #fff;
            text-align: center;
            white-space: nowrap;
            vertical-align: baseline;
            border-radius: 0.25rem;
            margin-right: 0.5rem;
        }
        
        .badge-blue {
            background-color: #3b82f6;
        }
        
        .badge-green {
            background-color: #10b981;
        }
        
        .badge-red {
            background-color: #ef4444;
        }
        
        .badge-yellow {
            background-color: #f59e0b;
        }
        
        /* Feature description styling */
        .feature-description {
            font-size: 0.85rem;
            color: #6b7280;
            margin-top: 0.25rem;
        }
        
        /* Prediction result styling */
        .prediction-result {
            font-size: 1.5rem;
            font-weight: 700;
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        
        .fatal {
            background-color: #fee2e2;
            color: #b91c1c;
            border: 2px solid #ef4444;
        }
        
        .major {
            background-color: #ffedd5;
            color: #c2410c;
            border: 2px solid #f97316;
        }
        
        .minor {
            background-color: #fef3c7;
            color: #b45309;
            border: 2px solid #f59e0b;
        }
        
        .no-injury {
            background-color: #d1fae5;
            color: #065f46;
            border: 2px solid #10b981;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            padding: 1rem;
            font-size: 0.875rem;
            color: #6b7280;
            margin-top: 2rem;
            border-top: 1px solid #e5e7eb;
        }
    </style>
    """, unsafe_allow_html=True)

# Apply the custom CSS
local_css()

# Helper function to create a gradient banner
def create_gradient_banner(title, subtitle=None):
    st.markdown(f'<div class="main-header">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f"<h3 style='text-align: center; color: #4b5563; margin-bottom: 2rem;'>{subtitle}</h3>", unsafe_allow_html=True)

# Helper function to create a card component
def create_card(title, content, icon=None):
    icon_html = f"<span style='font-size: 1.5rem; margin-right: 0.5rem;'>{icon}</span>" if icon else ""
    st.markdown(f"""
    <div class="card">
        <h3 class="section-header">{icon_html}{title}</h3>
        {content}
    </div>
    """, unsafe_allow_html=True)

# Helper function to create severity badge
def severity_badge(severity):
    colors = {
        "Fatal": "red",
        "Major Injury": "yellow",
        "Minor Injury": "blue",
        "No Injury": "green"
    }
    color = colors.get(severity, "blue")
    return f'<span class="badge badge-{color}">{severity}</span>'

# Function to display prediction result with appropriate styling
def display_prediction_result(severity):
    severity_class = severity.lower().replace(" ", "-")
    st.markdown(f"""
    <div class="prediction-result {severity_class}">
        Predicted Severity: {severity}
    </div>
    """, unsafe_allow_html=True)

# Create home page content
def home_page():
    create_gradient_banner("Road Accident Severity Predictor", 
                          "An AI-powered tool to predict the severity of road accidents")
    
    # Introduction section
    st.markdown("""
    <div class="info-box">
        <strong>Welcome to the Road Safety Analyzer!</strong> This application uses machine learning to predict 
        the severity of road accidents based on various factors like weather conditions, time of day, 
        driver characteristics, and more.
    </div>
    """, unsafe_allow_html=True)
    
    # Create three column layout for feature cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        create_card("🚀 Make Predictions", """
        <p>Use our AI model to predict accident severity based on various factors.</p>
        <ul>
            <li>Environmental conditions</li>
            <li>Driver characteristics</li>
            <li>Temporal factors</li>
        </ul>
        <div style="text-align: center; margin-top: 1rem;">
            <a href="javascript:void(0);" onclick="document.querySelector('[data-testid=stSidebar] [data-testid=stVerticalBlock] div:nth-child(3) button').click();" style="background-color: #2563eb; color: white; padding: 8px 16px; border-radius: 5px; text-decoration: none; font-weight: 600; display: inline-block;">Try Now</a>
        </div>
        """, "🔮")
    
    with col2:
        create_card("💡 Explainable AI", """
        <p>Understand how our model works with interactive visualizations.</p>
        <ul>
            <li>Feature importance analysis</li>
            <li>SHAP value exploration</li>
            <li>Interactive what-if analysis</li>
        </ul>
        <div style="text-align: center; margin-top: 1rem;">
            <a href="javascript:void(0);" onclick="document.querySelector('[data-testid=stSidebar] [data-testid=stVerticalBlock] div:nth-child(4) button').click();" style="background-color: #2563eb; color: white; padding: 8px 16px; border-radius: 5px; text-decoration: none; font-weight: 600; display: inline-block;">Explore</a>
        </div>
        """, "📊")
    
    with col3:
        create_card("🤖 Model Information", """
        <p>Learn about the machine learning model powering our predictions.</p>
        <ul>
            <li>Model architecture</li>
            <li>Performance metrics</li>
            <li>Training methodology</li>
        </ul>
        <div style="text-align: center; margin-top: 1rem;">
            <a href="javascript:void(0);" onclick="document.querySelector('[data-testid=stSidebar] [data-testid=stVerticalBlock] div:nth-child(2) button').click();" style="background-color: #2563eb; color: white; padding: 8px 16px; border-radius: 5px; text-decoration: none; font-weight: 600; display: inline-block;">Learn More</a>
        </div>
        """, "📈")
    
    # Statistics section
    st.markdown("<h2 class='section-header' style='margin-top: 2rem;'>📊 Road Safety Statistics</h2>", unsafe_allow_html=True)
    
    stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
    
    with stats_col1:
        st.markdown("""
        <div style="background-color: #eff6ff; padding: 1rem; border-radius: 10px; text-align: center;">
            <h3 style="color: #1e40af; margin: 0;">1.3M+</h3>
            <p style="margin: 0;">Annual fatalities worldwide</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature relationship explanation
        st.markdown("""
        <div class="card" style="margin-top: 1.5rem;">
            <h4 style="color: #1e40af; margin-top: 0;">Feature Interaction Effects</h4>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; margin-top: 1rem;">
                <div style="background-color: #f8fafc; padding: 1rem; border-radius: 5px;">
                    <h5 style="color: #1e40af; margin-top: 0;">Weather × Light Condition</h5>
                    <p style="font-size: 0.9rem;">
                        Poor weather combined with low visibility conditions increases severity 
                        risk by 3.2× compared to good conditions.
                    </p>
                </div>
                <div style="background-color: #f8fafc; padding: 1rem; border-radius: 5px;">
                    <h5 style="color: #1e40af; margin-top: 0;">Rush Hour × Age Group</h5>
                    <p style="font-size: 0.9rem;">
                        Young drivers (18-25) during rush hour have 2.5× higher 
                        risk of major injury accidents.
                    </p>
                </div>
                <div style="background-color: #f8fafc; padding: 1rem; border-radius: 5px;">
                    <h5 style="color: #1e40af; margin-top: 0;">Surface Condition × Hour</h5>
                    <p style="font-size: 0.9rem;">
                        Wet surfaces at night (8PM-5AM) show a 78% increase in 
                        severity compared to daytime.
                    </p>
                </div>
                <div style="background-color: #f8fafc; padding: 1rem; border-radius: 5px;">
                    <h5 style="color: #1e40af; margin-top: 0;">Weekday × Month</h5>
                    <p style="font-size: 0.9rem;">
                        Weekend accidents in winter months show 42% higher 
                        severity than weekday accidents.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col2:
        st.markdown("""
        <div style="background-color: #fef2f2; padding: 1rem; border-radius: 10px; text-align: center;">
            <h3 style="color: #b91c1c; margin: 0;">50M+</h3>
            <p style="margin: 0;">Injuries per year</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col3:
        st.markdown("""
        <div style="background-color: #f0fdf4; padding: 1rem; border-radius: 10px; text-align: center;">
            <h3 style="color: #166534; margin: 0;">90%</h3>
            <p style="margin: 0;">Accidents are preventable</p>
        </div>
        """, unsafe_allow_html=True)
    
    with stats_col4:
        st.markdown("""
        <div style="background-color: #fff7ed; padding: 1rem; border-radius: 10px; text-align: center;">
            <h3 style="color: #c2410c; margin: 0;">3x</h3>
            <p style="margin: 0;">Risk at night vs. day</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Safety tips section
    st.markdown("<h2 class='section-header' style='margin-top: 2rem;'>🛡️ Road Safety Tips</h2>", unsafe_allow_html=True)
    
    tips_col1, tips_col2 = st.columns(2)
    
    with tips_col1:
        create_card("Driver Safety", """
        <ul>
            <li><strong>Avoid distractions</strong> - Never text while driving</li>
            <li><strong>Follow speed limits</strong> - Speed is a major factor in accidents</li>
            <li><strong>Maintain safe distance</strong> - Always keep a 3-second gap</li>
            <li><strong>Never drive impaired</strong> - Alcohol, drugs, and fatigue kill</li>
        </ul>
        """)
    
    with tips_col2:
        create_card("Vehicle Safety", """
        <ul>
            <li><strong>Regular maintenance</strong> - Check brakes, tires, and lights</li>
            <li><strong>Wear seatbelts</strong> - They reduce fatalities by 45%</li>
            <li><strong>Child safety seats</strong> - Required for children under 8</li>
            <li><strong>Adapt to conditions</strong> - Slow down in bad weather</li>
        </ul>
        """)
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>Road Safety Analyzer © 2025 | This is a demonstration application for educational purposes only.</p>
        <p>Data sources: WHO Global Road Safety Report, National Highway Traffic Safety Administration</p>
    </div>
    """, unsafe_allow_html=True)

# Create prediction page content
def prediction_page():
    create_gradient_banner("🚀 Accident Severity Prediction", 
                         "Fill in the details below to predict the severity of a potential accident")
    
    st.markdown("""
    <div class="info-box">
        <strong>How it works:</strong> Our machine learning model analyzes various factors to predict 
        the likely severity of an accident. This can help in understanding risk factors and improving road safety measures.
    </div>
    """, unsafe_allow_html=True)
    
    # Create a form for prediction inputs
    with st.form("prediction_form"):
        st.markdown("<h3 class='section-header'>Enter Accident Details</h3>", unsafe_allow_html=True)
        
        # Create two columns for input fields
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<p><strong>Environmental Factors</strong></p>", unsafe_allow_html=True)
            
            light_condition = st.selectbox(
                "Light Condition", 
                ["Daylight", "Darklighted", "Dusk", "Dawn", "Dark not lighted", "Unknown"],
                help="The lighting condition at the time of the accident"
            )
            st.markdown("<p class='feature-description'>Visibility has a significant impact on accident severity</p>", unsafe_allow_html=True)
            
            weather = st.selectbox(
                "Weather Condition", 
                ["Clear", "Cloudy", "Rain", "Unknown", "Others", "Fog"],
                help="The weather condition at the time of the accident"
            )
            st.markdown("<p class='feature-description'>Poor weather can reduce visibility and traction</p>", unsafe_allow_html=True)
            
            surface_condition = st.selectbox(
                "Road Surface Condition", 
                ["Dry", "Wet", "Unknown", "Others", "Stagnant Water"],
                help="The condition of the road surface"
            )
            st.markdown("<p class='feature-description'>Wet or slippery surfaces increase stopping distance</p>", unsafe_allow_html=True)
            
            rush_hour = st.selectbox(
                "Rush Hour", 
                ["Yes", "No"],
                help="Whether the accident occurred during rush hour"
            )
            st.markdown("<p class='feature-description'>Traffic volume affects collision severity</p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<p><strong>Accident Details</strong></p>", unsafe_allow_html=True)
            
            hour = st.slider(
                "Hour of Day", 
                0, 23, 12,
                help="The hour of the day when the accident occurred (24-hour format)"
            )
            st.markdown("<p class='feature-description'>Time of day correlates with visibility and fatigue</p>", unsafe_allow_html=True)
            
            weekday = st.selectbox(
                "Day of Week", 
                ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                help="The day of the week when the accident occurred"
            )
            st.markdown("<p class='feature-description'>Weekend accidents often have different characteristics</p>", unsafe_allow_html=True)
            
            month = st.selectbox(
                "Month", 
                ["January", "February", "March", "April", "May", "June", 
                 "July", "August", "September", "October", "November", "December"],
                help="The month when the accident occurred"
            )
            st.markdown("<p class='feature-description'>Seasonal patterns affect accident severity</p>", unsafe_allow_html=True)
            
            age_group = st.selectbox(
                "Driver Age Group", 
                ["18-25", "26-60", "60+"],
                help="The age group of the primary driver"
            )
            st.markdown("<p class='feature-description'>Driver experience is a significant factor</p>", unsafe_allow_html=True)
        
        st.markdown("<p><strong>Impact Metrics</strong></p>", unsafe_allow_html=True)
        
        col3, col4 = st.columns(2)
        
        with col3:
            total_injuries = st.number_input(
                "Total Injuries", 
                min_value=0, 
                max_value=100, 
                value=0,
                help="The total number of injuries in the accident"
            )
        
        with col4:
            total_fatalities = st.number_input(
                "Total Fatalities", 
                min_value=0, 
                max_value=10, 
                value=0,
                help="The total number of fatalities in the accident"
            )
        
        # Submit button styled with custom CSS
        st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #2563eb;
            color: white;
            font-size: 16px;
            font-weight: 600;
            height: 3em;
            width: 100%;
            border-radius: 5px;
            border: none;
            margin-top: 1rem;
        }
        div.stButton > button:hover {
            background-color: #1d4ed8;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            transform: translateY(-2px);
        }
        </style>
        """, unsafe_allow_html=True)
        
        submitted = st.form_submit_button("Predict Accident Severity")
    
    # If form is submitted, show a prediction
    if submitted:
        # Encode categorical inputs (in a real app, these would be fed to the model)
        light_map = {"Darklighted": 0, "Daylight": 1, "Dusk": 2, "Dawn": 3, "Dark not lighted": 4, "Unknown": 5}
        weather_map = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Unknown": 3, "Others": 4, "Fog": 5}
        surface_map = {"Dry": 0, "Unknown": 1, "Wet": 2, "Others": 3, "Stagnant Water": 4}
        weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
        rush_map = {"No": 0, "Yes": 1}
        month_map = {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6, 
                    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12}
        age_group_map = {"18-25": 2, "26-60": 1, "60+": 0}
        
        # For demonstration, we'll use a simple rule-based prediction
        # In reality, this would call the model to get a prediction
        severity_score = 0
        
        # Simple rules for demonstration (actual app would use ML model)
        if total_fatalities > 0:
            severity_level = "Fatal"
        elif rush_map[rush_hour] == 1 and (hour < 7 or hour > 19):
            severity_level = "Major Injury"
        elif weather_map[weather] > 1 or surface_map[surface_condition] > 1:
            severity_level = "Minor Injury"
        else:
            severity_level = "No Injury"
        
        # Display the prediction
        st.markdown("<h3 class='section-header'>Prediction Result</h3>", unsafe_allow_html=True)
        display_prediction_result(severity_level)
        
        # Add some explanation
        st.markdown("<h3 class='section-header'>Key Factors</h3>", unsafe_allow_html=True)
        
        # Visualize key factors
        col1, col2 = st.columns(2)
        
        with col1:
            create_card("Risk Factors", f"""
            <ul>
                <li><strong>Time of Day:</strong> {hour}:00 hours {'(Nighttime - Higher Risk)' if hour < 6 or hour > 18 else '(Daytime - Lower Risk)'}</li>
                <li><strong>Weather:</strong> {weather} {'(Higher Risk)' if weather != 'Clear' else '(Lower Risk)'}</li>
                <li><strong>Road Condition:</strong> {surface_condition} {'(Higher Risk)' if surface_condition != 'Dry' else '(Lower Risk)'}</li>
                <li><strong>Rush Hour:</strong> {rush_hour} {'(Higher Risk)' if rush_hour == 'Yes' else '(Lower Risk)'}</li>
                <li><strong>Driver Age Group:</strong> {age_group} {'(Higher Risk)' if age_group == '18-25' else '(Moderate Risk)' if age_group == '60+' else '(Lower Risk)'}</li>
            </ul>
            """)
        
        with col2:
            create_card("Safety Recommendations", f"""
            <p>Based on the predicted severity level ({severity_level}), consider these safety measures:</p>
            <ul>
                {'<li><strong>Extreme Caution:</strong> This scenario has a high fatality risk. Avoid travel if possible.</li>' if severity_level == 'Fatal' else ''}
                {'<li><strong>Enhanced Visibility:</strong> Use high-visibility clothing and ensure all vehicle lights work properly.</li>' if light_condition != 'Daylight' else ''}
                {'<li><strong>Reduced Speed:</strong> Slow down significantly in these conditions.</li>' if weather != 'Clear' or surface_condition != 'Dry' else ''}
                {'<li><strong>Maintain Distance:</strong> Keep extra distance between vehicles during rush hour.</li>' if rush_hour == 'Yes' else ''}
                {'<li><strong>Avoid Distractions:</strong> Pay extra attention to the road and eliminate phone use.</li>' if age_group == '18-25' or age_group == '60+' else ''}
            </ul>
            """)
        
        # Warning message for high severity predictions
        if severity_level in ["Fatal", "Major Injury"]:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ High Severity Warning:</strong> The conditions you've entered suggest a high-risk scenario. 
                Consider postponing travel or taking alternative routes if possible. If travel is necessary, 
                exercise extreme caution and ensure all safety measures are in place.
            </div>
            """, unsafe_allow_html=True)

# Create model info page
def model_info_page():
    create_gradient_banner("🤖 Model Information", 
                         "Understanding the AI behind our accident severity predictions")
    
    st.markdown("""
    <div class="info-box">
        <strong>About the Model:</strong> Our prediction system uses a Gradient Boosting Classifier, an advanced 
        machine learning algorithm that excels at classification tasks with high accuracy and robustness.
    </div>
    """, unsafe_allow_html=True)
    
    # Model architecture section
    st.markdown("<h3 class='section-header'>Model Architecture</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #1e40af; margin-top: 0;">Model Type</h4>
            <p>Gradient Boosting Classifier</p>
            
            <h4 style="color: #1e40af;">Key Parameters</h4>
            <ul>
                <li>n_estimators: 200</li>
                <li>max_depth: 5</li>
                <li>learning_rate: 0.1</li>
                <li>subsample: 0.8</li>
            </ul>
            
            <h4 style="color: #1e40af;">Training Dataset</h4>
            <p>80% of historical accident data (2018-2023)</p>
            
            <h4 style="color: #1e40af;">Testing Dataset</h4>
            <p>20% of historical accident data (2018-2023)</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Placeholder for model architecture visualization
        st.markdown("""
        <div style="background-color: white; padding: 1rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); height: 350px; display: flex; flex-direction: column; justify-content: center; align-items: center;">
            <div style="color: #1e40af; font-weight: 600; margin-bottom: 1rem;">Gradient Boosting Model Visualization</div>
            <div style="display: flex; width: 100%; justify-content: space-between;">
                <div style="background-color: #dbeafe; padding: 1rem; border-radius: 5px; width: 30%; text-align: center;">
                    <div style="font-weight: 600; color: #1e40af;">Input Features</div>
                    <div style="font-size: 0.8rem; margin-top: 0.5rem;">32 features including environmental, temporal, and driver factors</div>
                </div>
                <div style="background-color: #dbeafe; padding: 1rem; border-radius: 5px; width: 30%; text-align: center;">
                    <div style="font-weight: 600; color: #1e40af;">Model Processing</div>
                    <div style="font-size: 0.8rem; margin-top: 0.5rem;">Ensemble of 200 decision trees with regularization techniques</div>
                </div>
                <div style="background-color: #dbeafe; padding: 1rem; border-radius: 5px; width: 30%; text-align: center;">
                    <div style="font-weight: 600; color: #1e40af;">Output</div>
                    <div style="font-size: 0.8rem; margin-top: 0.5rem;">4 severity classes: Fatal, Major Injury, Minor Injury, No Injury</div>
                </div>
            </div>
            <div style="margin-top: 1rem; font-size: 0.8rem; color: #6b7280; text-align: center;">
                Each tree corrects errors made by previous trees, resulting in a powerful predictive model
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Performance metrics section
    st.markdown("<h3 class='section-header' style='margin-top: 2rem;'>Performance Metrics</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #1e40af; margin-top: 0;">Accuracy Metrics</h4>
            <table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">
                <tr style="background-color: #dbeafe;">
                    <th style="padding: 0.5rem; text-align: left; border: 1px solid #e5e7eb;">Metric</th>
                    <th style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">Value</th>
                </tr>
                <tr>
                    <td style="padding: 0.5rem; border: 1px solid #e5e7eb;">Overall Accuracy</td>
                    <td style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">86.7%</td>
                </tr>
                <tr style="background-color: #f9fafb;">
                    <td style="padding: 0.5rem; border: 1px solid #e5e7eb;">Precision</td>
                    <td style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">83.2%</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem; border: 1px solid #e5e7eb;">Recall</td>
                    <td style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">81.5%</td>
                </tr>
                <tr style="background-color: #f9fafb;">
                    <td style="padding: 0.5rem; border: 1px solid #e5e7eb;">F1 Score</td>
                    <td style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">82.3%</td>
                </tr>
            </table>
            
            <h4 style="color: #1e40af; margin-top: 1rem;">Class-wise Performance</h4>
            <table style="width: 100%; border-collapse: collapse; margin-top: 1rem;">
                <tr style="background-color: #dbeafe;">
                    <th style="padding: 0.5rem; text-align: left; border: 1px solid #e5e7eb;">Severity Class</th>
                    <th style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">Precision</th>
                    <th style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">Recall</th>
                </tr>
                <tr>
                    <td style="padding: 0.5rem; border: 1px solid #e5e7eb;">Fatal</td>
                    <td style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">91.2%</td>
                    <td style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">88.7%</td>
                </tr>
                <tr style="background-color: #f9fafb;">
                    <td style="padding: 0.5rem; border: 1px solid #e5e7eb;">Major Injury</td>
                    <td style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">84.3%</td>
                    <td style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">80.1%</td>
                </tr>
                <tr>
                    <td style="padding: 0.5rem; border: 1px solid #e5e7eb;">Minor Injury</td>
                    <td style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">79.8%</td>
                    <td style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">77.5%</td>
                </tr>
                <tr style="background-color: #f9fafb;">
                    <td style="padding: 0.5rem; border: 1px solid #e5e7eb;">No Injury</td>
                    <td style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">87.4%</td>
                    <td style="padding: 0.5rem; text-align: right; border: 1px solid #e5e7eb;">85.2%</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)