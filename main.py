import streamlit as st
import pandas as pd

# Set Page Configuration
st.set_page_config(
    page_title="Accident Severity Predictor",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# Custom CSS for the main application
st.markdown("""
<style>
    /* Base styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Sidebar styling */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #F8FAFC;
    }
    
    /* Improve navigation bar styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background-color: #F1F5F9;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 0.375rem;
        padding: 0.75rem 1rem;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background-color: #3B82F6;
        color: white;
    }
    
    /* Style for navigation elements */
    [data-testid="stSidebarNav"] {
        background-color: #ffffff;
        border-radius: 0.5rem;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    [data-testid="stSidebarNav"] > div {
        padding: 0 !important;
    }
    
    [data-testid="stSidebarNav"] [data-testid="stVerticalBlock"] {
        padding: 0 !important;
    }
    
    [data-testid="stSidebarNav"] .css-fblp2m {
        padding: 0.5rem 1rem !important;
    }
    
    [data-testid="stSidebarNav"] [role="button"] {
        padding: 0.75rem 1rem !important;
        border-radius: 0 !important;
        transition: all 0.2s ease;
    }
    
    [data-testid="stSidebarNav"] [role="button"]:hover {
        background-color: #EFF6FF !important;
    }
    
    [data-testid="stSidebarNav"] [aria-selected="true"] {
        background-color: #DBEAFE !important;
        border-left: 4px solid #3B82F6 !important;
    }
    
    /* App title in sidebar */
    .sidebar-title {
        color: #1E3A8A;
        font-weight: 700;
        font-size: 1.25rem;
        padding: 1rem;
        text-align: center;
        border-bottom: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    
    /* Sidebar section */
    .sidebar-section {
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    .sidebar-section-title {
        color: #4B5563;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    
    /* Info card in sidebar */
    .sidebar-info-card {
        background-color: #EFF6FF;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .sidebar-info-title {
        color: #1E40AF;
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-info-content {
        color: #4B5563;
        font-size: 0.85rem;
        line-height: 1.5;
    }
    
    /* Footer in sidebar */
    .sidebar-footer {
        text-align: center;
        font-size: 0.85rem;
        color: #6B7280;
        padding: 1rem;
        border-top: 1px solid #E5E7EB;
        margin-top: 1rem;
    }
    
    /* Add a nice shadow to all inputs and selectboxes */
    .stTextInput, .stNumberInput, .stDateInput, .stSelectbox {
        filter: drop-shadow(0 1px 2px rgba(0,0,0,0.05));
    }
    
    /* Add animations to buttons */
    .stButton > button {
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Modify scrollbar for better look */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #F1F5F9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #CBD5E1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94A3B8;
    }
    
    /* Custom styling for navigation bar */
    div[data-testid="stNavigationContainer"] {
        background-color: #ffffff;
        border-radius: 0.5rem;
        padding: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Sidebar header with logo and app title
with st.sidebar:
    st.markdown("""
    <div class="sidebar-title">
        🚦 Accident Severity Predictor
    </div>
    """, unsafe_allow_html=True)
    
    # You can add additional sidebar elements here if needed
    st.markdown("""
    <div class="sidebar-info-card">
        <div class="sidebar-info-title">About This Application</div>
        <div class="sidebar-info-content">
            This tool uses machine learning to predict the severity of road accidents based on various environmental and driver-related factors.
        </div>
    </div>
    """, unsafe_allow_html=True)

# Create the navigation pages
home_page = st.Page("Apps/home.py", title="Home", icon="🏠")
prediction_page = st.Page("Apps/prediction.py", title="Make Prediction", icon="🚀")
explainer_page = st.Page("Apps/explainer.py", title="Explainer", icon="💡")

# Create a sidebar for navigation
pg = st.navigation(
    {"Navigation": [home_page, prediction_page, explainer_page]},
)

# Add footer to sidebar after navigation
with st.sidebar:
    st.markdown("""
    <div class="sidebar-footer">
        © 2025 | v1.0.3 | <a href="#" style="color: #3B82F6; text-decoration: none;">Help</a>
    </div>
    """, unsafe_allow_html=True)

# Run the navigation
pg.run()