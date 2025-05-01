import streamlit as st

# Custom CSS for styling the home page
st.markdown("""
<style>
    /* Base styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main header styling */
    .main-title {
        color: #1E3A8A;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    
    /* Section headers */
    .section-header {
        color: #2563EB;
        font-size: 1.5rem;
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #E5E7EB;
    }
    
    /* Info box styling */
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Warning box styling */
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1.5rem 0;
    }
    
    /* Step styling */
    .step-container {
        display: flex;
        margin-bottom: 1rem;
        align-items: flex-start;
    }
    
    .step-number {
        background-color: #3B82F6;
        color: white;
        font-weight: 600;
        width: 2rem;
        height: 2rem;
        border-radius: 9999px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-right: 1rem;
        flex-shrink: 0;
    }
    
    .step-content {
        padding-top: 0.25rem;
    }
    
    /* Horizontal divider */
    .divider {
        height: 1px;
        background-color: #E5E7EB;
        margin: 2rem 0;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #6B7280;
        font-size: 0.9rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E5E7EB;
    }
    
    /* Hero section styling */
    .hero-section {
        background: linear-gradient(to right, #EFF6FF, #DBEAFE);
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
    }
    
    .hero-title {
        color: #1E3A8A;
        font-size: 1.75rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .hero-subtitle {
        color: #3B82F6;
        font-size: 1.1rem;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    
    .hero-content {
        color: #4B5563;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">🚧 Road Traffic Accident Severity Prediction</h1>
    <p class="hero-subtitle">Leveraging machine learning to enhance road safety analysis</p>
    <p class="hero-content">
        Welcome to the <strong>Accident Severity Predictor</strong>! This intelligent application helps you 
        predict the severity of road traffic accidents based on environmental conditions, driver attributes, 
        and temporal patterns, providing valuable insights for safety planning.
    </p>
</div>
""", unsafe_allow_html=True)

# Key Features Section
st.markdown('<h2 class="section-header">Key Factors Analyzed</h2>', unsafe_allow_html=True)

# Using columns for the feature cards with FIXED HEIGHT via inline styles
col1, col2, col3 = st.columns(3)

# Each card will have a fixed height with inline style
card_style = """
    style="
        background-color: white; 
        border-radius: 0.5rem; 
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05), 0 10px 15px rgba(0, 0, 0, 0.03); 
        padding: 1.5rem; 
        height: 380px; 
        display: flex; 
        flex-direction: column;
    "
"""

with col1:
    st.markdown(f"""
    <div {card_style}>
        <h3 style="color: #1E40AF; font-size: 1.25rem; font-weight: 600; margin-bottom: 0.75rem;">🌦️ Environmental Factors</h3>
        <p style="color: #4B5563; font-size: 0.95rem; line-height: 1.5;">
            Analysis of weather conditions, lighting, and road surface states that can significantly impact accident outcomes.
        </p>
        <ul style="list-style-type: none; padding-left: 0; margin-left: 1.5rem; margin-top: auto;">
            <li style="position: relative; padding-left: 1.5rem; margin-bottom: 0.5rem; color: #4B5563;">
                <span style="content: '•'; color: #3B82F6; font-weight: bold; position: absolute; left: 0; top: 0;">•</span> 
                Weather conditions (rain, fog, clear)
            </li>
            <li style="position: relative; padding-left: 1.5rem; margin-bottom: 0.5rem; color: #4B5563;">
                <span style="content: '•'; color: #3B82F6; font-weight: bold; position: absolute; left: 0; top: 0;">•</span> 
                Light conditions (daylight, night)
            </li>
            <li style="position: relative; padding-left: 1.5rem; margin-bottom: 0.5rem; color: #4B5563;">
                <span style="content: '•'; color: #3B82F6; font-weight: bold; position: absolute; left: 0; top: 0;">•</span> 
                Road surface conditions (dry, wet, icy)
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div {card_style}>
        <h3 style="color: #1E40AF; font-size: 1.25rem; font-weight: 600; margin-bottom: 0.75rem;">👤 Driver & Vehicle Attributes</h3>
        <p style="color: #4B5563; font-size: 0.95rem; line-height: 1.5;">
            Evaluation of driver demographics and vehicle characteristics that correlate with accident severity.
        </p>
        <ul style="list-style-type: none; padding-left: 0; margin-left: 1.5rem; margin-top: auto;">
            <li style="position: relative; padding-left: 1.5rem; margin-bottom: 0.5rem; color: #4B5563;">
                <span style="content: '•'; color: #3B82F6; font-weight: bold; position: absolute; left: 0; top: 0;">•</span> 
                Driver age groups
            </li>
            <li style="position: relative; padding-left: 1.5rem; margin-bottom: 0.5rem; color: #4B5563;">
                <span style="content: '•'; color: #3B82F6; font-weight: bold; position: absolute; left: 0; top: 0;">•</span> 
                Vehicle types involved
            </li>
            <li style="position: relative; padding-left: 1.5rem; margin-bottom: 0.5rem; color: #4B5563;">
                <span style="content: '•'; color: #3B82F6; font-weight: bold; position: absolute; left: 0; top: 0;">•</span> 
                Driver behavior patterns
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div {card_style}>
        <h3 style="color: #1E40AF; font-size: 1.25rem; font-weight: 600; margin-bottom: 0.75rem;">⏰ Temporal Patterns</h3>
        <p style="color: #4B5563; font-size: 0.95rem; line-height: 1.5;">
            Identification of time-related factors that influence the severity of road accidents.
        </p>
        <ul style="list-style-type: none; padding-left: 0; margin-left: 1.5rem; margin-top: auto;">
            <li style="position: relative; padding-left: 1.5rem; margin-bottom: 0.5rem; color: #4B5563;">
                <span style="content: '•'; color: #3B82F6; font-weight: bold; position: absolute; left: 0; top: 0;">•</span> 
                Time of day (rush hours vs. off-peak)
            </li>
            <li style="position: relative; padding-left: 1.5rem; margin-bottom: 0.5rem; color: #4B5563;">
                <span style="content: '•'; color: #3B82F6; font-weight: bold; position: absolute; left: 0; top: 0;">•</span> 
                Day of week (weekday vs. weekend)
            </li>
            <li style="position: relative; padding-left: 1.5rem; margin-bottom: 0.5rem; color: #4B5563;">
                <span style="content: '•'; color: #3B82F6; font-weight: bold; position: absolute; left: 0; top: 0;">•</span> 
                Seasonal variations
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# How to Use Section
st.markdown('<h2 class="section-header"> How to Use</h2>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    This application provides multiple tools to help you understand and predict road accident severity. Follow these steps to make the most of its features.
</div>
""", unsafe_allow_html=True)

# Steps using the custom step styling
steps = [
    "Navigate to the sidebar and select your desired function",
    "Use **Make Prediction** to enter accident details and get severity predictions",
    "Explore the **Explainable AI** section to understand how different factors influence accident severity",
    "Review prediction results and explanations to gain insights for road safety improvements"
]

for i, step in enumerate(steps, 1):
    st.markdown(f"""
    <div class="step-container">
        <div class="step-number">{i}</div>
        <div class="step-content">{step}</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Model info with fixed height cards
col1, col2 = st.columns(2)

info_card_style = """
    style="
        background-color: #EFF6FF; 
        border-left: 4px solid #3B82F6; 
        border-radius: 0.375rem; 
        padding: 1rem; 
        height: 180px;
        display: flex;
        flex-direction: column;
    "
"""

with col1:
    st.markdown(f"""
    <div {info_card_style}>
        <h3 style="color: #1E40AF; margin-top: 0; font-size: 1.2rem;">Model Architecture</h3>
        <p style="color: #4B5563; font-size: 0.95rem; line-height: 1.5;">The prediction system uses a <strong>XGBoost Classifier</strong> trained on historical accident data. This model has an accuracy of <strong> 83.5% </strong></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div {info_card_style}>
        <h3 style="color: #1E40AF; margin-top: 0; font-size: 1.2rem;">Training Dataset</h3>
        <p style="color: #4B5563; font-size: 0.95rem; line-height: 1.5;">The model was trained on a comprehensive dataset of road accidents, featuring over 30 variables including environmental conditions, driver attributes, and accident characteristics.</p>
    </div>
    """, unsafe_allow_html=True)

# Warning notice
st.markdown("""
<div class="warning-box">
    <strong>⚠️ Disclaimer:</strong> All predictions are experimental and for demonstration purposes only. 
    The model should not be used as the sole basis for decision-making in real-world scenarios.
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 Road Accident Severity Predictor | Developed with ❤️ for road safety</p>
</div>
""", unsafe_allow_html=True)