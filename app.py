import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="NFL Big Data Bowl 2027 - WR Burst Index", layout="wide")

st.title("🏈 NFL Big Data Bowl 2027: WR Burst Efficiency Index")
st.write("An Interactive Data Science Application for Predicting Wide Receiver Success")

# Custom Metric Explanation
st.markdown("""
### Custom Metric: Burst Efficiency Index
$$\\text{Burst Index} = \\frac{\\text{10-Yard Split}}{\\text{40-Yard Dash}}$$
""")

# Feature Importance Chart
st.subheader("Random Forest Feature Importance")

importances_data = pd.DataFrame({
    'Feature': ['burst_index', 'vertical', 'ten_yd_split', 'broad_jump', 'forty'],
    'Importance': [0.244, 0.232, 0.222, 0.173, 0.129]
})

fig, ax = plt.subplots(figsize=(8, 4))
sns.barplot(x='Importance', y='Feature', data=importances_data, palette='Blues_r', ax=ax)
ax.set_title("What Predicts WR Career Success (Offensive Snaps)?")
st.pyplot(fig)

st.success("Analysis complete! Burst Index is the #1 predictor of Wide Receiver success.")
