

import streamlit as st
import pandas as pd



# Set Page Configuration
#st.set_page_config(page_title="Accident Severity Predictor",layout="wide",page_icon=":material/monitoring:")


# Create the navigation pages
home_page = st.Page("Apps/home.py", title="Home", icon="🏠")
prediction_page = st.Page("Apps/prediction.py", title = "Make Prediction", icon="🚀")
explainer_page = st.Page("Apps/explainer.py", title = "Another Explainer ", icon="💡")


# Create a sidebar for navigation
pg = st.navigation(
    {"Navigation Bar": [home_page, prediction_page, explainer_page]
     
     },
)

pg.run()
