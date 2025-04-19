import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your trained model
model = joblib.load('diabetes_xgboost_model.pkl')  # Change to your model path

# App title
st.title("🩺 Diabetes Risk Prediction Tool")

# Sidebar with user inputs
st.sidebar.header("Patient Parameters")


def user_input_features():
    age = st.sidebar.slider("Age", 20, 100, 45)
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
    glucose = st.sidebar.slider("Glucose Level", 50, 300, 100)
    activity = st.sidebar.selectbox("Activity Level", ["Low", "Moderate", "High"])
    family_history = st.sidebar.radio("Family History", ["No", "Yes"])
    smoker = st.sidebar.radio("Smoker", ["No", "Yes"])

    # Encode categorical features
    activity_map = {"Low": 0, "Moderate": 1, "High": 2}
    family_history_map = {"No": 0, "Yes": 1}
    smoker_map = {"No": 0, "Yes": 1}

    data = {
        'age': age,
        'bmi': bmi,
        'glucose_level': glucose,
        'physical_activity_level': activity_map[activity],
        'family_history': family_history_map[family_history],
        'smoker': smoker_map[smoker]
    }
    return pd.DataFrame(data, index=[0])


# Get user input
input_df = user_input_features()

# Display user inputs
st.subheader("Patient Input Parameters")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Display results
st.subheader("Prediction")
risk_status = np.array(["Low Risk", "High Risk"])
st.write(risk_status[prediction][0])

st.subheader("Prediction Probability")
st.write(f"High Risk Probability: {prediction_proba[0][1]:.2%}")
st.write(f"Low Risk Probability: {prediction_proba[0][0]:.2%}")

# Visual feedback
if prediction[0] == 1:
    st.error('⚠️ High diabetes risk detected! Consult a doctor.')
else:
    st.success('✅ Low diabetes risk detected! Maintain healthy habits.')

# Add some explainability
st.markdown("---")
st.subheader("How to Improve Your Score")
st.write("""
- Maintain BMI between 18.5-24.9
- Keep fasting glucose <100 mg/dL
- Exercise regularly (150+ mins/week)
- Avoid smoking
""")