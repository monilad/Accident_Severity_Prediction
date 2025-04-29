

import streamlit as st
import pandas as pd



# Set Page Configuration
#st.set_page_config(page_title="Accident Severity Predictor",layout="wide",page_icon=":material/monitoring:")


# Create the navigation pages
home_page = st.Page("Apps/home.py", title="Home", icon="🏠")
prediction_page = st.Page("Apps/prediction.py", title = "Make Prediction", icon="🚀")
explainer_page = st.Page("Apps/explainer.py", title = "Explainer ", icon="💡")


# Create a sidebar for navigation
pg = st.navigation(
    {"Navigation Bar": [home_page, prediction_page, explainer_page]
     
     },
)

pg.run()



# import streamlit as st
# import pandas as pd

# # Set Page Configuration
# st.set_page_config(
#     page_title="Accident Severity Predictor",
#     layout="wide",
#     page_icon="🚦"
# )

# # Create the navigation pages
# home_page = st.page("Apps/home.py", title="Home", icon="🏠") 
# model_page = st.page("Apps/models.py", title="Model Info", icon="🤖")
# prediction_page = st.page("Apps/prediction.py", title="Make Prediction", icon="🚀")
# interpret_page = st.page("Apps/interpret.py", title="Explainable AI", icon="💡")
# explainer_page = st.page("Apps/explainer.py", title="Another Explainer", icon="💡")
# finalexplainer_page = st.page("Apps/finalexplainer.py", title="Final Explainer", icon="💡")

# # Create a sidebar for navigation
# pg = st.navigation(
#     {
#         "Navigation Bar": [
#             home_page, 
#             model_page, 
#             prediction_page, 
#             interpret_page, 
#             explainer_page,
#             finalexplainer_page
#         ]
#     },
# )

# pg.run()