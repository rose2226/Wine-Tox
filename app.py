import streamlit as st # type: ignore
import numpy as np # type: ignore
import pandas as pd # type: ignore
import xgboost as xgb # type: ignore
import os
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore

# Attempt to load model
model_path = "models/xgboost_model.json"
model = None
try:
    if os.path.exists(model_path):
        model = xgb.Booster()
        model.load_model(model_path)
    else:
        st.error(f"Model file not found at '{model_path}'. Please ensure the model file exists in the correct directory.")
        st.stop()
except Exception as e:
    st.error(f"Failed to load model: {str(e)}")
    st.stop()

# Feature names and default values based on average wine measurements
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
    'density', 'pH', 'sulphates', 'alcohol'
]

# Default values (typical measurements)
default_values = {
    'fixed acidity': 7.2,
    'volatile acidity': 0.34,
    'citric acid': 0.32,
    'residual sugar': 2.6,
    'chlorides': 0.05,
    'free sulfur dioxide': 30.0,
    'total sulfur dioxide': 115.0,
    'density': 0.995,
    'pH': 3.3,
    'sulphates': 0.62,
    'alcohol': 10.5
}

# Input ranges for validation and sliders
input_ranges = {
    'fixed acidity': (3.8, 15.9, 0.1),       # (min, max, step)
    'volatile acidity': (0.08, 1.2, 0.01),
    'citric acid': (0.0, 1.0, 0.01),
    'residual sugar': (0.6, 15.5, 0.1),
    'chlorides': (0.01, 0.61, 0.01),
    'free sulfur dioxide': (1.0, 72.0, 1.0),
    'total sulfur dioxide': (6.0, 289.0, 1.0),
    'density': (0.9871, 1.0040, 0.0001),
    'pH': (2.7, 4.0, 0.01),
    'sulphates': (0.3, 2.0, 0.01),
    'alcohol': (8.0, 14.9, 0.1)
}

# Feature explanations
feature_explanations = {
    'fixed acidity': "Most acids involved with wine or fixed or nonvolatile (do not evaporate readily). Higher levels can give a tart taste.",
    'volatile acidity': "The amount of acetic acid in wine, which at too high of levels can lead to an unpleasant, vinegar taste.",
    'citric acid': "Found in small quantities, citric acid can add 'freshness' and flavor to wines.",
    'residual sugar': "The amount of sugar remaining after fermentation stops. Higher values create sweeter wine.",
    'chlorides': "The amount of salt in the wine. Higher levels can affect the taste negatively.",
    'free sulfur dioxide': "The free form of SO2 exists in equilibrium with molecular SO2. It prevents microbial growth and wine oxidation.",
    'total sulfur dioxide': "Amount of free and bound forms of SO2. In low concentrations, SO2 is mostly undetectable in wine, but at free SO2 concentrations over 50 ppm, SO2 becomes evident.",
    'density': "The density of water is close to that of water depending on the percent alcohol and sugar content.",
    'pH': "Describes how acidic or basic a wine is on a scale from 0 (very acidic) to 14 (very basic). Most wines are between 3-4 on the pH scale.",
    'sulphates': "A wine additive which can contribute to sulfur dioxide gas (S02) levels, which acts as an antimicrobial and antioxidant.",
    'alcohol': "The percent alcohol content of the wine. Higher alcohol content generally correlates with higher quality in red wines."
}

st.set_page_config(page_title="Wine Quality Predictor", layout="wide")

# --- Styles ---
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }
    .header-container {
        text-align: center;
        background: linear-gradient(135deg, #4c1d95, #6d28d9);
        padding: 40px 20px;
        border-radius: 16px;
        color: white;
        margin-bottom: 40px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    }
    .header-container h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .header-container p {
        font-size: 1.1rem;
        opacity: 0.9;
    }
    .form-container {
        background: #ffffff;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .form-container:hover {
        transform: translateY(-4px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
    }
    .input-label {
        font-weight: 600;
        font-size: 0.95rem;
        color: #334155;
        margin-bottom: 6px;
    }
    .input-description {
        font-size: 0.85rem;
        color: #64748b;
        margin-bottom: 10px;
        line-height: 1.4;
    }
    .stNumberInput input {
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 10px;
        font-size: 0.95rem;
    }
    
    /* Enhanced Button Styles */
    .stButton>button {
        background: linear-gradient(135deg, #4c1d95, #8b5cf6);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        padding: 12px 24px;
        border-radius: 10px;
        margin: 25px auto 0;
        display: block;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(109, 40, 217, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #6d28d9, #a78bfa);
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(109, 40, 217, 0.4);
    }
    
    .stButton>button:active {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(109, 40, 217, 0.4);
    }
    
    .stButton>button::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: 0.5s;
    }
    
    .stButton>button:hover::after {
        left: 100%;
    }
    
    /* Reset Defaults Button */
    .reset-defaults {
        background: #f1f5f9;
        color: #475569;
        font-size: 0.9rem;
        font-weight: 200;
        border: 1px solid #cbd5e1;
        padding: 16px 8px;
        border-radius: 10px;
        width: 60%;
        cursor: pointer;
        transition: all 0.2s ease;
        margin-top: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 5px;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
    }
    
    .reset-defaults:hover {
        background: #e2e8f0;
        color: #334155;
        border-color: #94a3b8;
        transform: translateY(-1px);
        box-shadow: 0 3px 7px rgba(0, 0, 0, 0.08);
    }
    
    .reset-defaults:active {
        transform: translateY(0);
    }
    
    /* Make Another Prediction Button */
    .another-button {
        background: linear-gradient(135deg, #ea580c, #fb923c);
        color: white;
        font-weight: 600;
        font-size: 1rem;
        border: none;
        padding: 12px 0;
        border-radius: 10px;
        width: 60%;
        margin: 25px auto 0;
        display: block;
        text-align: center;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(234, 88, 12, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .another-button:hover {
        background: linear-gradient(135deg, #f97316, #fdba74);
        transform: translateY(-3px);
        box-shadow: 0 6px 15px rgba(234, 88, 12, 0.4);
    }
    
    .another-button:active {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(234, 88, 12, 0.4);
    }
    
    .another-button::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
        transition: 0.5s;
    }
    
    .another-button:hover::after {
        left: 100%;
    }
    
    .results-section {
        background: #ffffff;
        padding: 40px;
        border-radius: 16px;
        box-shadow: 0 6px 24px rgba(0, 0, 0, 0.08);
        margin-top: 40px;
        animation: fadeIn 0.5s ease-in;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .result-good {
        background: linear-gradient(135deg, #065f46, #059669);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        border-left: 4px solid #064e3b;
    }
    .result-bad {
        background: linear-gradient(135deg, #991b1b, #dc2626);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        border-left: 4px solid #7f1d1d;
    }
    .result-good h2, .result-bad h2 {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .result-good h4, .result-bad h4 {
        font-size: 1.1rem;
        font-weight: 500;
        opacity: 0.9;
    }
    .feature-importance {
        margin-top: 30px;
        padding: 20px;
        background: #f8fafc;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }
    .info-tooltip {
        color: #6d28d9;
        font-size: 0.9rem;
        cursor: help;
    }
    .stTable table {
        background: #f8fafc;
        border-radius: 10px;
        overflow: hidden;
        border: none;
    }
    .stTable th {
        background: #e2e8f0;
        font-weight: 600;
        color: #334155;
    }
    .stTable td {
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.9rem;
    }
    .footer-container {
        margin-top: 60px;
        padding: 30px 0;
        background: linear-gradient(to right, #4c1d95, #6d28d9);
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        text-align: center;
        color: #f8fafc;
        position: relative;
        overflow: hidden;
    }
    
    .footer-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(to right, #f59e0b, #ef4444);
    }
    
    .footer-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        gap: 15px;
        position: relative;
        z-index: 1;
    }
    
    .footer-logo {
        font-weight: 700;
        font-size: 1.5rem;
        margin-bottom: 10px;
        letter-spacing: 1px;
    }
    
    .footer-tagline {
        font-size: 1rem;
        opacity: 0.9;
        margin-bottom: 15px;
        font-style: italic;
    }
    
    .footer-links {
        display: flex;
        gap: 20px;
        justify-content: center;
        margin-top: 10px;
    }
    
    .footer-link {
        color: #f8fafc;
        text-decoration: none;
        font-weight: 600;
        padding: 5px 10px;
        border-radius: 8px;
        transition: all 0.3s ease;
        background-color: rgba(255, 255, 255, 0.1);
    }
    
    .footer-link:hover {
        background-color: rgba(255, 255, 255, 0.2);
        transform: translateY(-3px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    .footer-copyright {
        font-size: 0.85rem;
        opacity: 0.8;
        margin-top: 15px;
    }
    
    .footer-icon {
        font-size: 1.2rem;
        margin-right: 5px;
        vertical-align: middle;
    }
    
    .wine-icon {
        animation: sway 3s ease-in-out infinite;
        display: inline-block;
        transform-origin: bottom center;
    }
    
    @keyframes sway {
        0%, 100% { transform: rotate(0deg); }
        50% { transform: rotate(5deg); }
    }
</style>
""", unsafe_allow_html=True)

# --- UI State Management ---
if "show_result" not in st.session_state:
    st.session_state["show_result"] = False

# Function to reset to default values
def reset_to_defaults():
    for feat in feature_names:
        st.session_state[feat] = default_values[feat]

# Extract feature importance from model
def get_feature_importance():
    importance_scores = model.get_score(importance_type='weight')
    # Convert to dataframe and sort
    importance_df = pd.DataFrame({
        'Feature': list(importance_scores.keys()),
        'Importance': list(importance_scores.values())
    })
    importance_df = importance_df.sort_values(by='Importance', ascending=False)
    return importance_df

# Plot feature importance
def plot_feature_importance(importance_df):
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=importance_df)
    plt.title('Feature Importance for Wine Quality Prediction', fontsize=16)
    plt.xlabel('Importance Score', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.tight_layout()
    return plt

# --- Form Input UI ---
if not st.session_state["show_result"]:
    st.markdown("""
    <div class="header-container">
        <h1>🍷 Wine Quality Predictor 🍇</h1>
        <p>Predict wine quality based on its chemical properties</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="form-container">', unsafe_allow_html=True)
    
    # Instructions and help
    with st.expander("ℹ️ About This Predictor"):
        st.write("""
        This application predicts wine quality based on its chemical properties. The model has been trained on a dataset of wine samples 
        and can classify wine as either 'Good' or 'Bad' quality. Adjust the sliders to input your wine's properties and see the prediction.
        
        The default values represent typical measurements for an average quality wine. Feel free to adjust them based on your specific wine's properties.
        """)
    
    # Add reset defaults button
    col1, col2 = st.columns([6, 1])
    with col2:
        st.button("Reset to Defaults", on_click=reset_to_defaults, key="reset_btn", type="primary")
    
    with st.form("prediction_form"):
        input_values = {}
        
        for i in range(0, len(feature_names), 2):
            col1, col2 = st.columns(2)
            
            with col1:
                feat1 = feature_names[i]
                label1 = feat1.replace("_", " ").title()
                
                st.markdown(f'<div class="input-label">{label1}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="input-description">{feature_explanations[feat1]}</div>', unsafe_allow_html=True)
                
                min_val, max_val, step = input_ranges[feat1]
                input_values[feat1] = st.slider(
                    label="",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_values[feat1]),
                    step=float(step),
                    key=feat1,
                    format="%.4f" if step < 0.1 else "%.1f"
                )
            
            if i + 1 < len(feature_names):
                with col2:
                    feat2 = feature_names[i + 1]
                    label2 = feat2.replace("_", " ").title()
                    
                    st.markdown(f'<div class="input-label">{label2}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="input-description">{feature_explanations[feat2]}</div>', unsafe_allow_html=True)
                    
                    min_val, max_val, step = input_ranges[feat2]
                    input_values[feat2] = st.slider(
                        label="",
                        min_value=float(min_val),
                        max_value=float(max_val),
                        value=float(default_values[feat2]),
                        step=float(step),
                        key=feat2,
                        format="%.4f" if step < 0.1 else "%.1f"
                    )

        submitted = st.form_submit_button("Predict Quality", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if submitted:
        st.session_state["input_values"] = input_values
        st.session_state["show_result"] = True
        st.rerun()

# --- Result Display UI ---
if st.session_state["show_result"]:
    input_values = st.session_state["input_values"]
    input_array = np.array([input_values[feat] for feat in feature_names]).reshape(1, -1)
    dmatrix = xgb.DMatrix(input_array, feature_names=feature_names)
    prediction = model.predict(dmatrix)[0]
    quality_label = "Good" if prediction >= 0.5 else "Bad"
    confidence_score = round(prediction * 10, 1)
    
    # Get feature importance for visualization
    importance_df = get_feature_importance()
    fig = plot_feature_importance(importance_df)

    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    st.markdown(f"""
        <div class="{'result-good' if quality_label == 'Good' else 'result-bad'}">
            <h2>{quality_label} Quality Wine</h2>
            <h4>Confidence Score: {confidence_score} / 10</h4>
        </div>
    """, unsafe_allow_html=True)

    # Display input properties and explanations
    st.subheader("🧪 Input Properties")
    
    property_data = []
    for feat in feature_names:
        property_data.append({
            "Property": feat.replace("_", " ").title(),
            "Value": input_values[feat],
            "Typical Range": f"{input_ranges[feat][0]} - {input_ranges[feat][1]}"
        })
    
    feature_df = pd.DataFrame(property_data)
    st.table(feature_df)
    
    # Feature importance visualization
    st.subheader("📊 What Makes Your Wine Good or Bad?")
    st.write("This chart shows which chemical properties have the most influence on wine quality prediction")
    st.pyplot(fig)
    
    # Interpretation of results
    st.subheader("🔍 Interpretation")
    top_features = importance_df.head(3)['Feature'].tolist()
    
    st.write(f"""
    Based on our model, the top {len(top_features)} factors influencing wine quality are: 
    {", ".join([f"**{f.replace('_', ' ').title()}**" for f in top_features])}.
    """)
    
    if quality_label == "Good":
        st.success(f"""
        Your wine is predicted to be of **Good Quality** with a confidence score of {confidence_score}/10. 
        This suggests it has a balanced combination of chemical properties that typically result in a pleasing taste profile.
        """)
    else:
        st.error(f"""
        Your wine is predicted to be of **Bad Quality** with a confidence score of {confidence_score}/10. 
        Consider adjusting some of the top influencing factors to improve quality.
        """)

    # Using the enhanced "another-button" style
    if st.button("Make Another Prediction", key="another", use_container_width=True):
        st.session_state["show_result"] = False
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
    
# --- Improved Footer ---
st.markdown("""
<div class="footer-container">
    <div class="footer-content">
        <div class="footer-logo">
            <span class="wine-icon">🍷</span> Wine Quality Predictor
        </div>
        <div class="footer-tagline">
            Made with grit, grind, and a splash of wine
        </div>
        <div class="footer-links">
            <a href="https://cranecloud.io" target="_blank" class="footer-link">
                <span class="footer-icon">🌍</span> Crane Cloud
            </a>
            <a href="#" class="footer-link">
                <span class="footer-icon">📊</span> Wine Dataset
            </a>
            <a href="#" class="footer-link">
                <span class="footer-icon">📚</span> Documentation
            </a>
        </div>
        <div class="footer-copyright">
            © 2025 Wine Quality Predictor | All rights reserved
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
