import streamlit as st
import pickle
import numpy as np

# Load model and scaler
model  = pickle.load(open("titanic_model.pkl",  "rb"))
scaler = pickle.load(open("titanic_scaler.pkl", "rb"))

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")
st.title("🚢 Titanic Survival Prediction App")
st.write("**Name:** Ahmad Ali | **Batch:** Data Science Weekday – Hyderabad")
st.markdown("---")
st.write("Enter passenger details to predict survival probability.")

col1, col2 = st.columns(2)

with col1:
    pclass  = st.selectbox("Passenger Class", [1, 2, 3], index=2)
    sex     = st.selectbox("Sex", ["Male", "Female"])
    age     = st.number_input("Age", min_value=1, max_value=100, value=30)
    sibsp   = st.number_input("Siblings/Spouses Aboard", min_value=0, max_value=10, value=0)

with col2:
    parch    = st.number_input("Parents/Children Aboard", min_value=0, max_value=10, value=0)
    fare     = st.number_input("Fare Paid", min_value=0.0, max_value=600.0, value=32.0, step=0.5)
    embarked = st.selectbox("Port of Embarkation", ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"])

st.markdown("---")

if st.button("🔍 Predict Survival"):
    sex_enc      = 1 if sex == "Female" else 0
    embarked_enc = {"Southampton (S)": 0, "Cherbourg (C)": 1, "Queenstown (Q)": 2}[embarked]

    input_data = np.array([[pclass, sex_enc, age, sibsp, parch, fare, embarked_enc]])
    input_scaled = scaler.transform(input_data)

    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    if prediction == 1:
        st.success(f"✅ Passenger Likely Survived!")
        st.write(f"**Survival Probability:** {probability*100:.1f}%")
    else:
        st.error(f"❌ Passenger Likely Did Not Survive")
        st.write(f"**Survival Probability:** {probability*100:.1f}%")

    st.markdown("---")
    st.caption("⚠️ This app is for educational purposes only.")
