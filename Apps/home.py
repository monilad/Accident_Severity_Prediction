
import streamlit as st

st.set_page_config(page_title="Road Accident Severity App", page_icon="🚧", layout="wide")

st.markdown("""
# 🚧 Road Traffic Accident Severity Prediction
Welcome to the **Accident Severity Predictor**! This application helps you **predict the severity of a road traffic accident** based on key conditions such as
- Environmental factors (weather, lighting, road surface)
- Driver and vehicle attributes
- Temporal patterns (time of day, day of week)

---

## 🛠 How to Use
👉 Navigate to the sidebar and select
- **Model Info** 🤖 – to view model summary and performance
- **Make Prediction** 🚀 – to try a real-time severity prediction
- **Explainable AI** 💡 – to explore how the model makes decisions

> ⚠️ All predictions are experimental and for demonstration purposes only.

---
""")
