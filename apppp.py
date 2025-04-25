import streamlit as st
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# Load XGBoost model
model = xgb.Booster()
model.load_model("models/xgboost_model.json")

# Features and defaults
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

default_values = {
    'fixed acidity': 7.4,
    'volatile acidity': 0.7,
    'citric acid': 0.0,
    'residual sugar': 1.9,
    'chlorides': 0.076,
    'free sulfur dioxide': 11.0,
    'total sulfur dioxide': 34.0,
    'density': 0.9978,
    'pH': 3.51,
    'sulphates': 0.56,
    'alcohol': 9.4
}

input_ranges = {
    'fixed acidity': (4.0, 15.0, 0.1),
    'volatile acidity': (0.1, 1.5, 0.01),
    'citric acid': (0.0, 1.0, 0.01),
    'residual sugar': (0.5, 15.0, 0.1),
    'chlorides': (0.01, 0.2, 0.001),
    'free sulfur dioxide': (1.0, 72.0, 1.0),
    'total sulfur dioxide': (6.0, 289.0, 1.0),
    'density': (0.9900, 1.0050, 0.0001),
    'pH': (2.5, 4.5, 0.01),
    'sulphates': (0.2, 1.2, 0.01),
    'alcohol': (8.0, 15.0, 0.1)
}

feature_explanations = {
    'fixed acidity': 'Determines tartness. Higher levels = sharper taste.',
    'volatile acidity': 'High levels may cause unpleasant vinegar taste.',
    'citric acid': 'Adds freshness and flavor. Naturally in grapes.',
    'residual sugar': 'Amount of natural sugar left after fermentation.',
    'chlorides': 'Influences wine’s salinity and microbial stability.',
    'free sulfur dioxide': 'Protects wine from oxidation and bacteria.',
    'total sulfur dioxide': 'Total SO2 added for preservation.',
    'density': 'Influenced by sugar/alcohol; indicates wine body.',
    'pH': 'Controls acidity, flavor stability, and freshness.',
    'sulphates': 'Enhances flavor and helps preservation.',
    'alcohol': 'Higher alcohol often means better quality perception.'
}

# Page and styling
st.set_page_config(page_title="Wine Quality Predictor", layout="wide")

st.markdown("""
<style>
h1, h2, h3 { color: #800000; }
.header-container { text-align: center; margin-bottom: 20px; }
.form-container {
    background-color: #fff;
    padding: 30px;
    border-radius: 12px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}
.input-label { font-weight: bold; margin-bottom: 5px; }
.input-description { font-size: 0.85rem; color: #666; margin-bottom: 10px; }
.results-section { margin-top: 40px; text-align: center; }
.result-good {
    background-color: #e6ffed; color: #007f00;
    padding: 20px; border-radius: 12px;
}
.result-bad {
    background-color: #ffe6e6; color: #b30000;
    padding: 20px; border-radius: 12px;
}
.footer-container {
    margin-top: 60px; text-align: center;
    font-size: 0.85rem; color: #888;
}
.footer-logo { font-weight: bold; font-size: 1.1rem; color: #800000; }
.footer-tagline { font-style: italic; }
.footer-link {
    margin: 0 10px; text-decoration: none; color: #800000;
}
</style>
""", unsafe_allow_html=True)

# Session state init
if "show_result" not in st.session_state:
    st.session_state["show_result"] = False

def reset_to_defaults():
    for feat in feature_names:
        st.session_state[feat] = default_values[feat]
    st.session_state["show_result"] = False

# Header
if not st.session_state["show_result"]:
    st.markdown('<div class="header-container"><h1>🍷 Wine Quality Predictor</h1><p>Input chemical properties to predict the quality of your wine.</p></div>', unsafe_allow_html=True)

# Form input
with st.form("input_form"):
    st.markdown('<div class="form-container">', unsafe_allow_html=True)

    cols = st.columns(2)
    for i, feat in enumerate(feature_names):
        col = cols[i % 2]
        with col:
            st.markdown(f'<div class="input-label">{feat.title()}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="input-description">{feature_explanations[feat]}</div>', unsafe_allow_html=True)
            min_val, max_val, step = input_ranges[feat]
            default = default_values[feat]
            value = st.session_state.get(feat, default)
            st.session_state[feat] = col.slider(
                f"{feat}", min_value=float(min_val), max_value=float(max_val),
                value=float(value), step=float(step), key=feat)

    submitted = st.form_submit_button("Predict Wine Quality")
    reset_clicked = st.form_submit_button("Reset to Default Values")

    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        st.session_state["show_result"] = True
    if reset_clicked:
        reset_to_defaults()

# Prediction output
if st.session_state["show_result"]:
    input_data = np.array([[st.session_state[feat] for feat in feature_names]])
    dmatrix = xgb.DMatrix(input_data, feature_names=feature_names)
    prediction = model.predict(dmatrix)[0]
    quality_label = "Good Quality 🍷" if prediction >= 6 else "Poor Quality 🧪"
    result_class = "result-good" if prediction >= 6 else "result-bad"

    st.markdown(f"""
        <div class="results-section">
            <div class="{result_class}">
                <h2>{quality_label}</h2>
                <h4>Predicted Score: {prediction:.2f}</h4>
            </div>
    """, unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    xgb.plot_importance(model, ax=ax, importance_type='gain', show_values=False, title='Feature Importance')
    ax.grid(False)
    ax.set_xlabel("Importance")
    st.pyplot(fig)

    if st.button("🔁 Make Another Prediction"):
        st.session_state["show_result"] = False

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer-container">
    <div class="footer-logo">🍇 WineVision AI</div>
    <div class="footer-tagline">Turning chemistry into quality 🍷</div>
    <div class="footer-links">
        <a href="#" class="footer-link">GitHub</a>
        <a href="#" class="footer-link">Contact</a>
        <a href="#" class="footer-link">Docs</a>
    </div>
    <div class="footer-copyright">© 2025 WineVision AI</div>
</div>
""", unsafe_allow_html=True)
