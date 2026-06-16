import streamlit as st
import pandas as pd
import joblib

# Load the saved AI
model = joblib.load('asd_model.pkl')

st.set_page_config(page_title="ASD Screening Tool", page_icon="🩺")

st.title("Pediatric ASD Screening Tool (AI-Powered)")
st.write("This tool uses a Machine Learning model (Random Forest) to detect ASD traits based on the Q-CHAT-10 criteria.")

st.sidebar.header("Demographics")
age = st.sidebar.slider("Age (Months)", 12, 36, 24)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
jaundice = st.sidebar.selectbox("Born with Jaundice?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
family = st.sidebar.selectbox("Family History of ASD?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
who = st.sidebar.selectbox("Who is completing this?", options=[0, 1, 2], format_func=lambda x: ["Health Pro", "Family Member", "Self"][x])
eth = st.sidebar.selectbox("Ethnicity Code (0-10)", options=list(range(11)))

st.header("Behavioral Observations")
q_labels = [
    "A1: Does your child look at you when you call their name?",
    "A2: How easy is it for you to get eye contact?",
    "A3: Does your child point to indicate they want something?",
    "A4: Does your child point to share interest (e.g. at a bird)?",
    "A5: Does your child pretend (e.g. care for a doll)?",
    "A6: Does your child follow where you are looking?",
    "A7: Does your child show signs of wanting to comfort you if upset?",
    "A8: Would you describe your child’s first words as atypical?",
    "A9: Does your child use simple gestures (e.g. waving)?",
    "A10: Does your child stare at nothing with no apparent purpose?"
]

answers = []
for i in range(10):
    ans = st.radio(q_labels[i], options=[0, 1], horizontal=True, key=f"q{i}")
    answers.append(ans)

if st.button("Analyze Results"):
    # Match the order of your training features
    input_data = pd.DataFrame([[
        answers[0], answers[1], answers[2], answers[3], answers[4], 
        answers[5], answers[6], answers[7], answers[8], answers[9],
        age, sex, eth, jaundice, family, who
    ]], columns=['A1','A2','A3','A4','A5','A6','A7','A8','A9','A10','Age_Mons','Sex','Ethnicity','Jaundice','Family_with_ASD','Who completed the test'])
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"ASD Traits Detected. Confidence: {probability:.2%}")
        st.write("Advice: Consider seeking professional consultation for a formal assessment.")
    else:
        st.success(f"No ASD Traits Detected. Confidence: {1-probability:.2%}")
        st.write("Advice: The screening results show no immediate cause for concern.")

st.info("Disclaimer: This is a screening tool, not a clinical diagnosis.")