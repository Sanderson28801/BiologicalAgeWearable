import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import xgboost as xgb

st.write("Welcome to your Biological Age")

model = xgb.XGBRegressor()
model.load_model("toy_xgb_model.json")

chrono_input = st.slider("Chronological Age", 18, 60, 21)
rhr_input = st.slider("Resting Heart Rate (RHR)", 40, 100, 65)
hrv_input = st.slider("Heart Rate Variability (HRV)", 20, 100, 55)
sleep_input = st.slider("Sleep Score", 0, 100, 80)

input_data = pd.DataFrame([{
    'chronological_age': chrono_input,
    'rhr': rhr_input,
    'hrv': hrv_input,
    'sleep_score': sleep_input
}])

labels = ["RHR", "VO2 Max", "HRV", "BMI"]
values = [20, 40, 30, 10]
predicted_bio_age = int(model.predict(input_data)[0].round(0))
figure = px.pie(
    names=labels,
    values=values,
    hole=0.6,
    title="Your Biological Age",
)

figure.update_layout(annotations=[
    dict(text=f"{predicted_bio_age} years",
         x=0.5,
         y=0.5,
         font_size=20,
         showarrow=False)
])

st.plotly_chart(figure, use_container_width=True)
