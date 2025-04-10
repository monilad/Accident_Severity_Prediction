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

# Set page config - only needed if this page is running standalone
if "navigation" not in st.session_state:
    st.set_page_config(
        page_title="Final Accident Severity Explainer",
        page_icon="💡",
        layout="wide"
    )

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
        st.error(f"Error loading model or data: {str(e)}")
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

def show_feature_importance(model):
    """Display feature importance from the model"""
    st.subheader("Feature Importance")
    
    if model is None:
        st.error("Model is not loaded. Cannot show feature importance.")
        return
    
    # Get feature names and descriptions
    feature_names = get_feature_names()
    feature_descriptions = get_feature_descriptions()
    
    try:
        importances = model.feature_importances_
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Create a DataFrame for the feature importance
        importance_df = pd.DataFrame({
            'Feature': [feature_names[i] for i in indices],
            'Importance': [importances[i] for i in indices],
            'Description': [feature_descriptions.get(feature_names[i], '') for i in indices]
        })
        
        # Display the top 15 features
        top_features = importance_df.head(15)
        
        # Create a horizontal bar chart with plotly
        fig = px.bar(
            top_features,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top 15 Features by Importance',
            color='Importance',
            color_continuous_scale='Viridis'
        )
        
        # Add feature descriptions as hover text
        fig.update_traces(
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<br>Description: %{text}<extra></extra>',
            text=top_features['Description']
        )
        
        fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Show the data table
        with st.expander("View all feature importance data"):
            st.dataframe(importance_df)
            
    except Exception as e:
        st.error(f"Error calculating feature importance: {str(e)}")
        st.code(traceback.format_exc())

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
    impact_df['Color'] = impact_df['Impact'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
    
    # Create the horizontal bar chart
    fig = px.bar(
        impact_df,
        x='Impact',
        y='Feature',
        orientation='h',
        color='Color',
        color_discrete_map={'Positive': 'green', 'Negative': 'red'},
        title=f'Top 10 Features by Impact on Prediction ({severity_classes[class_idx]})'
    )
    
    # Improve layout
    fig.update_layout(height=500, xaxis_title="SHAP Impact Value", yaxis_title="Feature")
    
    return fig

def show_shap_analysis(model, sample_data, scaler):
    """Display SHAP values for model interpretability with updates for SHAP v0.20+"""
    st.subheader("SHAP Value Analysis")
    
    if model is None:
        st.error("Model is not loaded. Cannot perform SHAP analysis.")
        return
    
    if sample_data is None:
        st.warning("Sample data not available for SHAP analysis. Please upload a dataset.")
        
        # Option to upload data
        uploaded_file = st.file_uploader("Upload sample data for SHAP analysis (CSV)", type=['csv'])
        if uploaded_file is not None:
            try:
                sample_data = pd.read_csv(uploaded_file)
                st.success("Data uploaded successfully!")
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
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
    
    # Analysis button    
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
                st.write(f"### SHAP Summary Plot")
                st.write("This plot shows how each feature impacts the model prediction across all samples.")
                
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
                    st.error(f"Error creating summary plot: {str(e)}")
                    st.code(traceback.format_exc())
                
                # Force plot for a specific sample
                if sample_size > 0:
                    st.write(f"### Sample {sample_index} Force Plot for {severity_classes[class_idx]}")
                    
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
                        st.error(f"Error in force plot: {str(e)}")
                        
                        # Create a fallback visualization
                        st.write("#### Feature Impact Chart (Alternative to Force Plot)")
                        fig = create_feature_impact_chart(feature_names, shap_values, class_idx, sample_index, severity_classes)
                        st.plotly_chart(fig)
                
                # Waterfall plot for the selected sample and class
                if sample_size > 0:
                    st.write(f"### Waterfall Plot for Sample {sample_index}, {severity_classes[class_idx]}")
                    
                    try:
                        # Use smaller figure size for waterfall plot
                        plt.figure(figsize=(8, 6))
                        
                        # Get sample values for waterfall plot
                        if len(shap_values.shape) == 3:  # Multi-class
                            # For waterfall plot in multi-class, we need to use the specific format
                            shap.plots.waterfall(shap_values[sample_index, :, class_idx], max_display=8)
                        else:
                            shap.plots.waterfall(shap_values[sample_index], max_display=8)
                            
                        st.pyplot(plt)
                        plt.clf()  # Clear the figure
                        
                    except Exception as e:
                        st.error(f"Error generating waterfall plot: {str(e)}")
                        
                        # Fallback visualization
                        st.write("#### Feature Impact Chart (Alternative to Waterfall Plot)")
                        # We already created this above, so we don't need to create it again
                
            except Exception as e:
                st.error(f"Error in SHAP analysis: {str(e)}")
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
                st.write(f"### Sample {sample_index} Force Plot for {severity_classes[class_idx]}")
                
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
                    st.write(f"### Waterfall Plot for Sample {sample_index}, {severity_classes[class_idx]}")
                    plt.figure(figsize=(10, 8))
                    
                    # Get sample values for waterfall plot
                    if len(shap_values.shape) == 3:  # Multi-class
                        shap.plots.waterfall(shap_values[sample_index, :, class_idx], max_display=10)
                    else:
                        shap.plots.waterfall(shap_values[sample_index], max_display=10)
                        
                    st.pyplot(plt)
                    plt.clf()  # Clear the figure
                    
                except Exception as e:
                    st.error(f"Error showing explanation: {str(e)}")
                    
                    # Create a fallback visualization
                    st.write("#### Feature Impact Chart (Alternative Visualization)")
                    fig = create_feature_impact_chart(feature_names, shap_values, class_idx, sample_index, severity_classes)
                    st.plotly_chart(fig)
                
            except Exception as e:
                st.error(f"Error showing explanation: {str(e)}")
                st.code(traceback.format_exc())

def show_interactive_explanation(model, scaler):
    """Show an interactive explanation interface where users can input values and see predictions"""
    st.subheader("Interactive Feature Analysis")
    
    if model is None:
        st.error("Model is not loaded. Cannot perform interactive analysis.")
        return
    
    try:
        st.write("Adjust the features below to see how they affect the prediction:")
        
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
            
            # Display prediction
            st.success(f"Predicted Severity: **{severity}**")
            
            # Show probability distribution
            prob_df = pd.DataFrame({
                'Severity': list(severity_map.values()),
                'Probability': probabilities
            })
            
            fig = px.bar(
                prob_df,
                x='Severity',
                y='Probability',
                title='Prediction Probability Distribution',
                color='Severity',
                color_discrete_sequence=px.colors.qualitative.Set1
            )
            st.plotly_chart(fig)
            
            # Calculate SHAP values for this input
            try:
                explainer = shap.Explainer(model)
                shap_values = explainer(scaled_input)
                
                # Show SHAP values for each class
                st.write("### Feature Contribution to Prediction")
                
                # Tabs for different classes
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
                            st.write(f"#### Force Plot for {severity_map[i]}")
                            
                            try:
                                plt.figure(figsize=(12, 4))
                                
                                # Use the newer SHAP force plot API
                                shap.plots.force(
                                    base_value, 
                                    class_shap_values, 
                                    feature_names=get_feature_names(),
                                    matplotlib=True,
                                    show=False
                                )
                                st.pyplot(plt)
                                plt.clf()  # Clear figure
                            except Exception as e:
                                st.error(f"Force plot error: {str(e)}")
                                
                                # Fallback visualization
                                impact_df = pd.DataFrame({
                                    'Feature': get_feature_names(),
                                    'Impact': class_shap_values
                                })
                                impact_df['AbsImpact'] = abs(impact_df['Impact'])
                                impact_df = impact_df.sort_values('AbsImpact', ascending=False).head(10)
                                impact_df['Color'] = impact_df['Impact'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
                                
                                fig = px.bar(
                                    impact_df,
                                    x='Impact',
                                    y='Feature',
                                    orientation='h',
                                    color='Color',
                                    color_discrete_map={'Positive': 'green', 'Negative': 'red'},
                                    title=f'Top 10 Features Impact on {severity_map[i]} Prediction'
                                )
                                st.plotly_chart(fig)
                            
                            # Waterfall plot
                            st.write(f"#### Waterfall Plot for {severity_map[i]}")
                            try:
                                # Smaller figure size for interactive waterfall plot
                                plt.figure(figsize=(8, 6))
                                
                                # Use the newer SHAP waterfall plot API with fewer features displayed
                                if len(shap_values.shape) == 3:  # Multi-class
                                    shap.plots.waterfall(shap_values[0, :, i], max_display=8)
                                else:
                                    shap.plots.waterfall(shap_values[0], max_display=8)
                                    
                                st.pyplot(plt)
                                plt.clf()  # Clear figure
                            except Exception as e:
                                st.error(f"Waterfall plot error: {str(e)}")
                                # Fallback already handled above
                                
                        except Exception as e:
                            st.error(f"Error showing explanation for class {i}: {str(e)}")
                
            except Exception as e:
                st.error(f"Error generating SHAP explanation: {str(e)}")
                st.code(traceback.format_exc())
    
    except Exception as e:
        st.error(f"Error in interactive explanation: {str(e)}")
        st.code(traceback.format_exc())

def main():
    st.title("🔍 Final Explainable AI for Accident Severity Prediction")
    st.write("""
    This tool helps you understand how our machine learning model makes predictions about accident severity.
    Use the tabs below to explore different aspects of the model's decision-making process.
    """)
    
    # Load the model and data
    model, scaler, sample_data = load_models_and_data()
    
    if model is None:
        st.error("Failed to load the model. Please check that the model files (best_xgb_model.pkl and scaler.pkl) are available in the same directory as this script.")
        st.info("You can upload model files below:")
        
        model_file = st.file_uploader("Upload model file (best_xgb_model.pkl)", type=['pkl'])
        scaler_file = st.file_uploader("Upload scaler file (scaler.pkl)", type=['pkl'])
        
        if model_file and scaler_file:
            try:
                # Save the uploaded files
                with open('best_xgb_model.pkl', 'wb') as f:
                    f.write(model_file.getbuffer())
                with open('scaler.pkl', 'wb') as f:
                    f.write(scaler_file.getbuffer())
                    
                st.success("Files uploaded successfully! Please refresh the page.")
                model, scaler, _ = load_models_and_data()
            except Exception as e:
                st.error(f"Error saving files: {str(e)}")
        
        return
    
    # Create tabs for different explanation methods
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "SHAP Analysis", "Interactive Explanation"])
    
    with tab1:
        show_feature_importance(model)
    
    with tab2:
        show_shap_analysis(model, sample_data, scaler)
    
    with tab3:
        show_interactive_explanation(model, scaler)
    
    # Add explanatory information at the bottom
    st.markdown("""
    ### Understanding Model Interpretability
    
    - **Feature Importance**: Shows which features have the greatest impact on predictions across all data.
    - **SHAP Analysis**: Shows how each feature contributes to individual predictions using game theory principles.
    - **Interactive Explanation**: Lets you explore how changing input values affects predictions.
    
    These tools help make the 'black box' of machine learning more transparent and understandable.
    """)

if __name__ == "__main__":
    main()