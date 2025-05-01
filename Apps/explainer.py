import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap
import os
import traceback

# Custom CSS for explainer page
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
    
    /* Card styling */
    .card-container {
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
    
    /* Detail box styling */
    .detail-box {
        background-color: #F9FAFB;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Section title */
    .section-title {
        color: #4B5563;
        font-weight: 600;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.75rem;
    }
    
    /* Feature info styling */
    .feature-info {
        display: flex;
        margin-bottom: 0.5rem;
    }
    
    .feature-name {
        width: 150px;
        font-size: 0.9rem;
        font-weight: 500;
        color: #4B5563;
    }
    
    .feature-value {
        flex-grow: 1;
        font-size: 0.9rem;
        color: #1F2937;
    }
    
    /* Tab container styling */
    .stTabs {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Custom tab styling */
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
    
    /* Button styling */
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
    
    /* Chart container */
    .chart-container {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    /* Legend styling */
    .legend-container {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        margin-bottom: 1rem;
    }
    
    .legend-item {
        display: flex;
        align-items: center;
    }
    
    .legend-color {
        width: 12px;
        height: 12px;
        border-radius: 9999px;
        margin-right: 0.5rem;
    }
    
    .legend-label {
        font-size: 0.85rem;
        color: #4B5563;
    }
    
    /* Definition container */
    .definition-box {
        background-color: #F3F4F6;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-top: 1.5rem;
    }
    
    .definition-title {
        color: #4B5563;
        font-weight: 600;
        font-size: 1rem;
        margin-bottom: 0.5rem;
    }
    
    .definition-content {
        color: #6B7280;
        font-size: 0.9rem;
        line-height: 1.5;
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
    
    /* Error message styling */
    .error-box {
        background-color: #FEE2E2;
        border-left: 4px solid #DC2626;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    
    .success-box {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        border-radius: 0.375rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Improve file uploader styling */
    [data-testid="stFileUploader"] {
        background-color: #F9FAFB;
        border-radius: 0.5rem;
        padding: 1.5rem;
        border: 1px dashed #D1D5DB;
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

# ORIGINAL FUNCTIONS FROM YOUR CODE
def load_models_and_data():
    """Load the model, scaler, and sample data for explanations"""
    try:
        model = joblib.load('best_xgb_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Try to load sample data
        sample_data = None
        if os.path.exists('df_sample_new.csv'):
            sample_data = pd.read_csv('df_sample_new.csv')
        
        return model, scaler, sample_data
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <p style="margin: 0; color: #991B1B;"><strong>Error loading model or data:</strong> {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
        return None, None, None

def get_feature_names():
    """Get feature names used in the model"""
    return [
        'Year', 'Distance', 'Totalinjuries', 'Totalfatalities', 'Collisionmanner',
        'Lightcondition', 'Weather', 'SurfaceCondition', 'Unittype_One',
        'Gender_Drv1', 'Traveldirection_One', 'AlcoholUse_Drv1', 'DrugUse_Drv1',
        'Unittype_Two', 'Gender_Drv2', 'Traveldirection_Two', 'AlcoholUse_Drv2',
        'DrugUse_Drv2', 'Hour', 'Weekday', 'Month', 'Weekend', 'Rush_Hour',
        'Hazardous_Road', 'Age_Group_Drv1', 'Age_Group_Drv2', 'Substance_Use',
        'Junction_Category', 'Violation_Category_Drv1', 'Violation_Category_Drv2',
        'Unitaction_Category_Two', 'Unitaction_Category_One'
    ]

def get_feature_descriptions():
    """Return descriptions for each feature for better interpretability"""
    return {
        'Year': 'Year of accident',
        'Distance': 'Distance from reference point',
        'Totalinjuries': 'Total number of injuries',
        'Totalfatalities': 'Total number of fatalities',
        'Collisionmanner': 'Manner of collision',
        'Lightcondition': 'Light conditions at accident site',
        'Weather': 'Weather conditions',
        'SurfaceCondition': 'Road surface condition',
        'Unittype_One': 'Vehicle type of first driver',
        'Gender_Drv1': 'Gender of first driver',
        'Traveldirection_One': 'Travel direction of first vehicle',
        'AlcoholUse_Drv1': 'Alcohol use by first driver',
        'DrugUse_Drv1': 'Drug use by first driver',
        'Unittype_Two': 'Vehicle type of second driver',
        'Gender_Drv2': 'Gender of second driver',
        'Traveldirection_Two': 'Travel direction of second vehicle',
        'AlcoholUse_Drv2': 'Alcohol use by second driver',
        'DrugUse_Drv2': 'Drug use by second driver',
        'Hour': 'Hour of the day',
        'Weekday': 'Day of the week',
        'Month': 'Month of the year',
        'Weekend': 'Whether accident occurred on weekend',
        'Rush_Hour': 'Whether accident occurred during rush hour',
        'Hazardous_Road': 'Whether road was hazardous',
        'Age_Group_Drv1': 'Age group of first driver',
        'Age_Group_Drv2': 'Age group of second driver',
        'Substance_Use': 'Whether substance use was involved',
        'Junction_Category': 'Type of road junction',
        'Violation_Category_Drv1': 'Violation category of first driver',
        'Violation_Category_Drv2': 'Violation category of second driver',
        'Unitaction_Category_Two': 'Action category of second vehicle',
        'Unitaction_Category_One': 'Action category of first vehicle'
    }

def create_feature_impact_chart(feature_names, shap_values, class_idx, sample_idx, severity_classes):
    """Create a fallback visualization when SHAP plots fail"""
    if len(shap_values.shape) == 3:  # Multi-class
        sample_values = shap_values.values[sample_idx, :, class_idx]
    else:  # Binary or single output
        sample_values = shap_values.values[sample_idx, :]
        
    # Create a DataFrame with feature impacts
    impact_df = pd.DataFrame({
        'Feature': feature_names,
        'Impact': sample_values
    })
    
    # Calculate absolute impact for sorting
    impact_df['AbsImpact'] = abs(impact_df['Impact'])
    
    # Sort and take top 10 features
    impact_df = impact_df.sort_values('AbsImpact', ascending=False).head(10)
    
    # Add color based on positive/negative impact
    impact_df['Color'] = impact_df['Impact'].apply(lambda x: 'Positive Impact' if x > 0 else 'Negative Impact')
    
    # Create the horizontal bar chart with improved styling
    fig = px.bar(
        impact_df,
        x='Impact',
        y='Feature',
        orientation='h',
        color='Color',
        color_discrete_map={'Positive Impact': '#10B981', 'Negative Impact': '#EF4444'},
        labels={'x': 'SHAP Impact Value', 'y': 'Feature'},
        template="plotly_white"
    )
    
    # Improve layout
    fig.update_layout(
        title=f'Top 10 Features by Impact on {severity_classes[class_idx]} Prediction',
        title_x=0.5,
        height=500,
        xaxis_title="Impact on Prediction",
        yaxis_title="Feature",
        legend_title=None,
        font=dict(family="Inter, sans-serif", size=12)
    )
    
    # Add a vertical line at x=0
    fig.add_shape(
        type='line',
        x0=0, y0=-0.5,
        x1=0, y1=9.5,
        line=dict(color='gray', width=1, dash='dash')
    )
    
    return fig

def show_shap_analysis(model, sample_data, scaler):
    """Display SHAP values for model interpretability with updates for SHAP v0.20+"""
    st.markdown('<h2 class="sub-header">SHAP Value Analysis</h2>', unsafe_allow_html=True)
    
    if model is None:
        st.markdown("""
        <div class="error-box">
            <p style="margin: 0; color: #991B1B;">Model is not loaded. Cannot perform SHAP analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    if sample_data is None:
        st.markdown("""
        <div class="warning-box">
            <p style="margin: 0; color: #92400E;">Sample data not available for SHAP analysis. Please upload a dataset.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Option to upload data
        uploaded_file = st.file_uploader("Upload sample data for SHAP analysis (CSV)", type=['csv'])
        if uploaded_file is not None:
            try:
                sample_data = pd.read_csv(uploaded_file)
                st.markdown("""
                <div class="success-box">
                    <p style="margin: 0; color: #065F46;">Data uploaded successfully!</p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <p style="margin: 0; color: #991B1B;"><strong>Error loading data:</strong> {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
                return
        else:
            return
    
    # Create placeholders outside the button execution
    feature_names = get_feature_names()
    
    # Initialize session state for storing computation results
    if 'shap_calculated' not in st.session_state:
        st.session_state.shap_calculated = False
    
    # Define severity class labels
    severity_classes = ["Fatal", "Major Injury", "Minor Injury", "No Injury"]
    
    # Class selection - allow user to select which class to explain
    selected_class = st.selectbox(
        "Select severity class to explain:",
        [f"Class {i} ({severity_classes[i]})" for i in range(len(severity_classes))],
        index=0
    )
    class_idx = int(selected_class.split()[1][0])  # Extract class number
    
    # Sample selection - always show this if we have data
    sample_size = min(10, len(sample_data)) if sample_data is not None else 0
    if sample_size > 0:
        sample_index = st.selectbox("Select sample to explain:", range(sample_size))
        
        # Show sample details with improved styling
        st.markdown('<div class="detail-box">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Sample Details</div>', unsafe_allow_html=True)
        
        # Get a subset of relevant features to display
        key_features = [
            'Totalinjuries', 'Totalfatalities', 'Lightcondition', 'Weather', 
            'SurfaceCondition', 'Hour', 'Weekday', 'Month', 'Rush_Hour', 'Age_Group_Drv1'
        ]
        
        # Create a mapping for categorical features to make them more readable
        light_map_rev = {0: "Darklighted", 1: "Daylight", 2: "Dusk", 3: "Dawn", 4: "Dark not lighted", 5: "Unknown"}
        weather_map_rev = {0: "Clear", 1: "Cloudy", 2: "Rain", 3: "Unknown", 4: "Others", 5: "Fog"}
        surface_map_rev = {0: "Dry", 1: "Unknown", 2: "Wet", 3: "Others", 4: "Stagnant Water"}
        weekday_map_rev = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        rush_map_rev = {0: "No", 1: "Yes"}
        month_map_rev = {1: "January", 2: "February", 3: "March", 4: "April", 5: "May", 6: "June", 
                       7: "July", 8: "August", 9: "September", 10: "October", 11: "November", 12: "December"}
        age_group_map_rev = {2: "18-25", 1: "26-60", 0: "60+"}
        
        # Display feature values
        for feature in key_features:
            feature_value = sample_data.iloc[sample_index][feature]
            
            # Map categorical features to human-readable values
            if feature == 'Lightcondition' and feature_value in light_map_rev:
                display_value = light_map_rev[feature_value]
            elif feature == 'Weather' and feature_value in weather_map_rev:
                display_value = weather_map_rev[feature_value]
            elif feature == 'SurfaceCondition' and feature_value in surface_map_rev:
                display_value = surface_map_rev[feature_value]
            elif feature == 'Weekday' and feature_value in weekday_map_rev:
                display_value = weekday_map_rev[feature_value]
            elif feature == 'Rush_Hour' and feature_value in rush_map_rev:
                display_value = rush_map_rev[feature_value]
            elif feature == 'Month' and feature_value in month_map_rev:
                display_value = month_map_rev[feature_value]
            elif feature == 'Age_Group_Drv1' and feature_value in age_group_map_rev:
                display_value = age_group_map_rev[feature_value]
            elif feature == 'Hour':
                display_value = f"{int(feature_value)}:00"
            else:
                display_value = feature_value
            
            # Display feature name and value
            st.markdown(f"""
            <div class="feature-info">
                <div class="feature-name">{get_feature_descriptions().get(feature, feature)}</div>
                <div class="feature-value">{display_value}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Analysis button with improved styling
    if st.button("Generate SHAP Analysis"):
        with st.spinner("Calculating SHAP values... This may take a moment."):
            try:
                # Scale the data
                scaled_data = scaler.transform(sample_data)
                
                # Create a SHAP explainer
                explainer = shap.Explainer(model)
                shap_values = explainer(scaled_data)
                
                # Store in session state
                st.session_state.explainer = explainer
                st.session_state.shap_values = shap_values
                st.session_state.scaled_data = scaled_data
                st.session_state.class_idx = class_idx
                st.session_state.shap_calculated = True
                
                # SHAP summary plot showing impact for each class
                st.markdown('<h3 class="sub-header">SHAP Summary Plot</h3>', unsafe_allow_html=True)
                st.markdown("""
                <div class="info-box">
                    <p style="margin: 0;">This plot shows how each feature impacts the model prediction across all samples.</p>
                </div>
                """, unsafe_allow_html=True)
                
                try:
                    # Create the summary plot with a smaller figure size
                    plt.figure(figsize=(8, 6))
                    
                    # Use class_names parameter to show the different classes in the legend
                    # And use the plot_type='bar' to show the mean absolute SHAP values by class
                    shap.summary_plot(
                        shap_values.values, 
                        scaled_data, 
                        feature_names=feature_names,
                        class_names=severity_classes,
                        plot_type='bar',
                        show=False
                    )
                    
                    st.pyplot(plt)
                    plt.clf()  # Clear the figure for next plot
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-box">
                        <p style="margin: 0; color: #991B1B;"><strong>Error creating summary plot:</strong> {str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.code(traceback.format_exc())
                
                # Force plot for a specific sample
                if sample_size > 0:
                    st.markdown(f'<h3 class="sub-header">Sample {sample_index} Force Plot for {severity_classes[class_idx]}</h3>', unsafe_allow_html=True)
                    
                    try:
                        # Get expected value(s)
                        expected_value = explainer.expected_value
                        
                        # Handle different formats of expected_value
                        if isinstance(expected_value, np.ndarray) or isinstance(expected_value, list):
                            base_value = expected_value[class_idx]
                        else:
                            base_value = expected_value
                        
                        # Get sample SHAP values for the selected class
                        if len(shap_values.shape) == 3:  # Multi-class format
                            sample_shap_values = shap_values.values[sample_index, :, class_idx]
                        else:
                            sample_shap_values = shap_values.values[sample_index, :]
                        
                        # Use shap.plots.force for newer SHAP versions (v0.20+)
                        plt.figure(figsize=(12, 4))
                        shap.plots.force(
                            base_value,
                            sample_shap_values,
                            feature_names=feature_names,
                            matplotlib=True,
                            show=False
                        )
                        st.pyplot(plt)
                        plt.clf()  # Clear the figure
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-box">
                            <p style="margin: 0; color: #991B1B;"><strong>Error in force plot:</strong> {str(e)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Create a fallback visualization
                        st.markdown('<h4 class="sub-header">Feature Impact Chart (Alternative to Force Plot)</h4>', unsafe_allow_html=True)
                        fig = create_feature_impact_chart(feature_names, shap_values, class_idx, sample_index, severity_classes)
                        st.plotly_chart(fig)
                
                # Waterfall plot for the selected sample and class
                if sample_size > 0:
                    st.markdown(f'<h3 class="sub-header">Waterfall Plot for Sample {sample_index}, {severity_classes[class_idx]}</h3>', unsafe_allow_html=True)
                    
                    try:
                        # Use smaller figure size for waterfall plot
                        plt.figure(figsize=(8, 6))
                        
                        # Get sample values for waterfall plot
                        if len(shap_values.shape) == 3:  # Multi-class
                            # For waterfall plot in multi-class, we need to be careful with feature names
                            # Create a modified Explanation object with named features
                            explanation = shap_values[sample_index, :, class_idx]
                            # Modify the feature names directly in the explanation object
                            explanation.feature_names = feature_names
                            # Now plot with the modified explanation
                            shap.plots.waterfall(
                                explanation, 
                                max_display=8,
                                show=False
                            )
                        else:
                            # Create a modified Explanation object with named features
                            explanation = shap_values[sample_index]
                            # Modify the feature names directly in the explanation object
                            explanation.feature_names = feature_names
                            # Now plot with the modified explanation
                            shap.plots.waterfall(
                                explanation, 
                                max_display=8,
                                show=False
                            )
                            
                        st.pyplot(plt)
                        plt.clf()  # Clear the figure
                        
                    except Exception as e:
                        st.markdown(f"""
                        <div class="error-box">
                            <p style="margin: 0; color: #991B1B;"><strong>Error generating waterfall plot:</strong> {str(e)}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Fallback visualization
                        st.markdown('<h4 class="sub-header">Feature Impact Chart (Alternative to Waterfall Plot)</h4>', unsafe_allow_html=True)
                        # We already created this above, so we don't need to create it again
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <p style="margin: 0; color: #991B1B;"><strong>Error in SHAP analysis:</strong> {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
                st.code(traceback.format_exc())
    
    # If SHAP values have been calculated previously and we have a valid sample index
    elif st.session_state.shap_calculated and sample_size > 0:
        if st.button("Show explanation for selected sample"):
            try:
                # Get stored values from session state
                explainer = st.session_state.explainer
                shap_values = st.session_state.shap_values
                scaled_data = st.session_state.scaled_data
                
                # Handle the expected_value properly
                expected_value = explainer.expected_value
                
                # Force plot
                st.markdown(f'<h3 class="sub-header">Sample {sample_index} Force Plot for {severity_classes[class_idx]}</h3>', unsafe_allow_html=True)
                
                try:
                    # Handle different formats of expected_value
                    if isinstance(expected_value, np.ndarray) or isinstance(expected_value, list):
                        base_value = expected_value[class_idx]
                    else:
                        base_value = expected_value
                    
                    # Get sample SHAP values for the selected class
                    if len(shap_values.shape) == 3:  # Multi-class format
                        sample_shap_values = shap_values.values[sample_index, :, class_idx]
                    else:
                        sample_shap_values = shap_values.values[sample_index, :]
                    
                    # Use shap.plots.force for newer SHAP versions
                    plt.figure(figsize=(12, 4))
                    shap.plots.force(
                        base_value,
                        sample_shap_values,
                        feature_names=feature_names,
                        matplotlib=True,
                        show=False
                    )
                    st.pyplot(plt)
                    plt.clf()  # Clear the figure
                    
                    # Waterfall plot
                    st.markdown(f'<h3 class="sub-header">Waterfall Plot for Sample {sample_index}, {severity_classes[class_idx]}</h3>', unsafe_allow_html=True)
                    plt.figure(figsize=(10, 8))
                    
                    # Get sample values for waterfall plot
                    if len(shap_values.shape) == 3:  # Multi-class
                        # Create a modified Explanation object with named features
                        explanation = shap_values[sample_index, :, class_idx]
                        # Modify the feature names directly in the explanation object
                        explanation.feature_names = feature_names
                        # Now plot with the modified explanation
                        shap.plots.waterfall(
                            explanation, 
                            max_display=10,
                            show=False
                        )
                    else:
                        # Create a modified Explanation object with named features
                        explanation = shap_values[sample_index]
                        # Modify the feature names directly in the explanation object
                        explanation.feature_names = feature_names
                        # Now plot with the modified explanation
                        shap.plots.waterfall(
                            explanation, 
                            max_display=10,
                            show=False
                        )
                        
                    st.pyplot(plt)
                    plt.clf()  # Clear the figure
                    
                except Exception as e:
                    st.markdown(f"""
                    <div class="error-box">
                        <p style="margin: 0; color: #991B1B;"><strong>Error showing explanation:</strong> {str(e)}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Create a fallback visualization
                    st.markdown('<h4 class="sub-header">Feature Impact Chart (Alternative Visualization)</h4>', unsafe_allow_html=True)
                    fig = create_feature_impact_chart(feature_names, shap_values, class_idx, sample_index, severity_classes)
                    st.plotly_chart(fig)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <p style="margin: 0; color: #991B1B;"><strong>Error showing explanation:</strong> {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
                st.code(traceback.format_exc())

    # Add explanatory information about SHAP values
    st.markdown("""
    <div class="definition-box">
        <div class="definition-title">How to Interpret SHAP Values</div>
        <div class="definition-content">
            <p>SHAP values explain the impact of each feature on a single prediction:</p>
            <ul>
                <li><strong>Positive values (green/right)</strong> push the prediction toward a particular severity class</li>
                <li><strong>Negative values (red/left)</strong> push the prediction away from a particular severity class</li>
                <li>The <strong>base value</strong> represents the average model output over the training dataset</li>
                <li>The <strong>sum of all SHAP values</strong> plus the base value equals the model's prediction for that instance</li>
            </ul>
            <p>SHAP values are based on game theory and provide consistent, locally accurate feature attributions.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_interactive_explanation(model, scaler):
    """Show an interactive explanation interface where users can input values and see predictions"""
    st.markdown('<h2 class="sub-header">Interactive Feature Analysis</h2>', unsafe_allow_html=True)
    
    if model is None:
        st.markdown("""
        <div class="error-box">
            <p style="margin: 0; color: #991B1B;">Model is not loaded. Cannot perform interactive analysis.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    try:
        st.markdown("""
        <div class="info-box">
            <p style="margin: 0;">Adjust the features below to see how they affect the prediction:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a card container for the interactive controls
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        
        # Create columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            injuries = st.slider("Total Injuries", 0, 10, 1)
            fatalities = st.slider("Total Fatalities", 0, 5, 0)
            light_condition = st.selectbox("Light Condition", 
                                         ["Daylight", "Darklighted", "Dusk", "Dawn", "Dark not lighted", "Unknown"],
                                         index=0)
            weather = st.selectbox("Weather", 
                                 ["Clear", "Cloudy", "Rain", "Unknown", "Others", "Fog"],
                                 index=0)
        
        with col2:
            hour = st.slider("Hour of Day", 0, 23, 12)
            age_group = st.selectbox("Age Group of Driver 1", 
                                    ["18-25", "26-60", "60+"],
                                    index=1)
            rush_hour = st.selectbox("Rush Hour", ["Yes", "No"], index=1)
            surface_condition = st.selectbox("Surface Condition", 
                                           ["Dry", "Wet", "Unknown", "Others", "Stagnant Water"],
                                           index=0)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Mappings
        light_map = {"Darklighted": 0, "Daylight": 1, "Dusk": 2, "Dawn": 3, "Dark not lighted": 4, "Unknown": 5}
        weather_map = {"Clear": 0, "Cloudy": 1, "Rain": 2, "Unknown": 3, "Others": 4, "Fog": 5}
        surface_map = {"Dry": 0, "Unknown": 1, "Wet": 2, "Others": 3, "Stagnant Water": 4}
        rush_map = {"No": 0, "Yes": 1}
        age_group_map = {"18-25": 2, "26-60": 1, "60+": 0}
        
        # Create a default input dictionary with all required features
        input_dict = {
            'Year': 2025,
            'Distance': 0,
            'Totalinjuries': injuries,
            'Totalfatalities': fatalities,
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
            'Weekday': 3,  # Thursday
            'Month': 6,  # June
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
        
        if st.button("Calculate Prediction and Explanation"):
            # Create DataFrame from input
            input_df = pd.DataFrame([input_dict])
            
            # Scale input
            scaled_input = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(scaled_input)[0]
            probabilities = model.predict_proba(scaled_input)[0]
            
            # Map prediction to severity label
            severity_map = {0: "Fatal", 1: "Major Injury", 2: "Minor Injury", 3: "No Injury"}
            severity = severity_map[prediction]
            
            # Display prediction with nice styling
            st.markdown(f"""
            <div class="success-box">
                <p style="margin: 0; color: #065F46; font-size: 1.2rem; font-weight: 600;">Predicted Severity: <span style="font-size: 1.4rem;">{severity}</span></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show probability distribution with improved styling
            st.markdown('<h3 class="sub-header">Prediction Probability Distribution</h3>', unsafe_allow_html=True)
            
            # Create color map for severity classes
            color_map = {
                "Fatal": "#DC2626",
                "Major Injury": "#F59E0B",
                "Minor Injury": "#2563EB",
                "No Injury": "#10B981"
            }
            
            # Create a better-looking chart
            fig = px.bar(
                x=list(severity_map.values()),
                y=probabilities,
                color=list(severity_map.values()),
                color_discrete_map=color_map,
                labels={'x': 'Severity Level', 'y': 'Probability'},
                template="plotly_white"
            )
            
            fig.update_layout(
                title="Probability Distribution by Severity Class",
                title_x=0.5,
                xaxis_title="Severity Level",
                yaxis_title="Probability",
                yaxis_range=[0, 1],
                showlegend=False,
                height=400,
                font=dict(family="Inter, sans-serif")
            )
            
            # Add value labels on top of bars
            fig.update_traces(
                texttemplate='%{y:.1%}',
                textposition='outside'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate SHAP values for this input
            try:
                explainer = shap.Explainer(model)
                shap_values = explainer(scaled_input)
                
                # Get feature names for proper labeling
                feature_names = get_feature_names()
                
                # Show SHAP values for each class
                st.markdown('<h3 class="sub-header">Feature Contribution to Prediction</h3>', unsafe_allow_html=True)
                
                # Tabs for different classes with improved styling
                class_tabs = st.tabs([f"Class {i} ({severity_map[i]})" for i in range(len(severity_map))])
                
                for i, tab in enumerate(class_tabs):
                    with tab:
                        try:
                            # Handle the expected_value for multi-class
                            expected_value = explainer.expected_value
                            if isinstance(expected_value, np.ndarray) or isinstance(expected_value, list):
                                base_value = expected_value[i]
                            else:
                                base_value = expected_value
                            
                            # Get appropriate SHAP values
                            if len(shap_values.shape) == 3:  # Multi-class
                                class_shap_values = shap_values.values[0, :, i]
                            else:
                                class_shap_values = shap_values.values[0, :]
                            
                            # Force plot
                            st.markdown(f'<h4 class="sub-header">Force Plot for {severity_map[i]}</h4>', unsafe_allow_html=True)
                            
                            try:
                                plt.figure(figsize=(12, 4))
                                
                                # Use the newer SHAP force plot API with explicit feature names
                                shap.plots.force(
                                    base_value, 
                                    class_shap_values, 
                                    feature_names=feature_names,  # Explicitly pass feature names
                                    matplotlib=True,
                                    show=False
                                )
                                st.pyplot(plt)
                                plt.clf()  # Clear figure

                          
                            except Exception as e:
                                st.markdown(f"""
                                <div class="error-box">
                                    <p style="margin: 0; color: #991B1B;"><strong>Force plot error:</strong> {str(e)}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Fallback visualization with improved styling
                                impact_df = pd.DataFrame({
                                    'Feature': feature_names,
                                    'Impact': class_shap_values
                                })
                                impact_df['AbsImpact'] = abs(impact_df['Impact'])
                                impact_df = impact_df.sort_values('AbsImpact', ascending=False).head(10)
                                impact_df['Color'] = impact_df['Impact'].apply(lambda x: 'Positive Impact' if x > 0 else 'Negative Impact')
                                
                                fig = px.bar(
                                    impact_df,
                                    x='Impact',
                                    y='Feature',
                                    orientation='h',
                                    color='Color',
                                    color_discrete_map={'Positive Impact': '#10B981', 'Negative Impact': '#EF4444'},
                                    title=f'Top 10 Features Impact on {severity_map[i]} Prediction'
                                )
                                
                                fig.update_layout(
                                    height=500,
                                    xaxis_title="Impact Value",
                                    yaxis_title="Feature",
                                    font=dict(family="Inter, sans-serif")
                                )
                                
                                st.plotly_chart(fig)
                            
                            # Waterfall plot
                            st.markdown(f'<h4 class="sub-header">Waterfall Plot for {severity_map[i]}</h4>', unsafe_allow_html=True)
                            try:
                                # Smaller figure size for interactive waterfall plot
                                plt.figure(figsize=(8, 6))
                                
                                # Use the newer SHAP waterfall plot API with modified explanation object
                                if len(shap_values.shape) == 3:  # Multi-class
                                    # Create a modified Explanation object with named features
                                    explanation = shap_values[0, :, i]
                                    # Modify the feature names directly in the explanation object
                                    explanation.feature_names = feature_names
                                    # Now plot with the modified explanation
                                    shap.plots.waterfall(
                                        explanation, 
                                        max_display=8,
                                        show=False
                                    )
                                else:
                                    # Create a modified Explanation object with named features
                                    explanation = shap_values[0]
                                    # Modify the feature names directly in the explanation object
                                    explanation.feature_names = feature_names
                                    # Now plot with the modified explanation
                                    shap.plots.waterfall(
                                        explanation, 
                                        max_display=8,
                                        show=False
                                    )
                                                                       
                                st.pyplot(plt)
                                plt.clf()  # Clear figure
                            except Exception as e:
                                st.markdown(f"""
                                <div class="error-box">
                                    <p style="margin: 0; color: #991B1B;"><strong>Waterfall plot error:</strong> {str(e)}</p>
                                </div>
                                """, unsafe_allow_html=True)
                                # Fallback already handled above
                                
                        except Exception as e:
                            st.markdown(f"""
                            <div class="error-box">
                                <p style="margin: 0; color: #991B1B;"><strong>Error showing explanation for class {i}:</strong> {str(e)}</p>
                            </div>
                            """, unsafe_allow_html=True)
                
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <p style="margin: 0; color: #991B1B;"><strong>Error generating SHAP explanation:</strong> {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
                st.code(traceback.format_exc())
    
    except Exception as e:
        st.markdown(f"""
        <div class="error-box">
            <p style="margin: 0; color: #991B1B;"><strong>Error in interactive explanation:</strong> {str(e)}</p>
        </div>
        """, unsafe_allow_html=True)
        st.code(traceback.format_exc())

def main():
    st.markdown('<h1 class="main-header">🔍 Explainable AI for Accident Severity Prediction</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>This tool helps you understand how our machine learning model makes predictions about accident severity.
        Use the tabs below to explore different aspects of the model's decision-making process.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load the model and data
    model, scaler, sample_data = load_models_and_data()
    
    if model is None:
        st.markdown("""
        <div class="error-box">
            <p style="margin: 0; color: #991B1B;"><strong>Failed to load the model.</strong> Please check that the model files (best_xgb_model.pkl and scaler.pkl) are available in the same directory as this script.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <p style="margin: 0;">You can upload model files below:</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a card container for the file uploads
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        
        model_file = st.file_uploader("Upload model file (best_xgb_model.pkl)", type=['pkl'])
        scaler_file = st.file_uploader("Upload scaler file (scaler.pkl)", type=['pkl'])
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if model_file and scaler_file:
            try:
                # Save the uploaded files
                with open('best_xgb_model.pkl', 'wb') as f:
                    f.write(model_file.getbuffer())
                with open('scaler.pkl', 'wb') as f:
                    f.write(scaler_file.getbuffer())
                    
                st.markdown("""
                <div class="success-box">
                    <p style="margin: 0; color: #065F46;"><strong>Files uploaded successfully!</strong> Please refresh the page.</p>
                </div>
                """, unsafe_allow_html=True)
                model, scaler, _ = load_models_and_data()
            except Exception as e:
                st.markdown(f"""
                <div class="error-box">
                    <p style="margin: 0; color: #991B1B;"><strong>Error saving files:</strong> {str(e)}</p>
                </div>
                """, unsafe_allow_html=True)
        
        return
    
    # Create tabs for different explanation methods with improved styling
    tab1, tab2 = st.tabs(["SHAP Analysis", "Interactive Explanation"])
    
    with tab1:
        show_shap_analysis(model, sample_data, scaler)
    
    with tab2:
        show_interactive_explanation(model, scaler)
    
    # Add explanatory information at the bottom with improved styling
    st.markdown("""
    <div class="definition-box">
        <div class="definition-title">Understanding Model Interpretability</div>
        <div class="definition-content">
            <ul>
                <li><strong>Feature Importance:</strong> Shows which features have the greatest impact on predictions across all data.</li>
                <li><strong>SHAP Analysis:</strong> Shows how each feature contributes to individual predictions using game theory principles.</li>
                <li><strong>Interactive Explanation:</strong> Lets you explore how changing input values affects predictions.</li>
            </ul>
            <p>These tools help make the 'black box' of machine learning more transparent and understandable.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add footer
    st.markdown("""
    <div class="footer">
        <p>Explainable AI Dashboard | Developed for understanding accident severity predictions</p>
        <p style="font-size: 0.8rem; margin-top: 0.5rem;">© 2025 Road Safety Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()