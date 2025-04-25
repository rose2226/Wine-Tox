# Wine-Tox
 Wine-Tox is Wine Quality Prediction App. The repository contains a Streamlit web application that predicts the quality of red wine based on various features using an XGBoost machine learning model.

Overview The application allows users to input wine characteristics (e.g., acidity, sugar content, alcohol level) and receive a prediction of the wine's quality ("Good" or "Bad") along with a quality score (0-10). The quality score represents the model's confidence in its prediction.

Files

app.py: The main Streamlit application script.

xgboost_model.json: The saved XGBoost model file.

wine quality_red.csv: The dataset used to train the model and fit the scaler.

README.md: This file. Prerequisites

Python 3.6 or higher

pip (Python package installer)
