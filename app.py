import streamlit as st
import pandas as pd
import joblib

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
pos_milestone_questions = {
    "A1": "Does your child look at you when you call their name?",
    "A2": "How easy is it for you to get eye contact with your child?",
    "A3": "Does your child point to indicate that s/he wants something?",
    "A4": "Does your child point to share interest/enjoyment with you?",
    "A5": "Does your child pretend? (e.g., caring for a doll, talking on a toy phone?)",
    "A6": "Does your child follow where you are looking?",
    "A7": "If you are visibly upset, does your child show signs of wanting to comfort you?",
    "A9": "Does your child use simple gestures (e.g., waving goodbye)?"
}

for key, text in pos_milestone_questions.items():
    ans = st.radio(text, options=["Yes", "No"], horizontal=True, key=key)
    # Mapping: Yes -> 0 (Typical), No -> 1 (Autistic Trait)
    processed_answers[key] = 1 if ans == "No" else 0

ans_a8 = st.radio("A8: Would you describe your child's first words as typical or atypical?", 
                  options=["Typical", "Atypical/Delayed"], horizontal=True, key="A8")
processed_answers["A8"] = 1 if ans_a8 == "Atypical/Delayed" else 0

ans_a10 = st.radio("A10: Does your child stare at nothing with no apparent purpose or have unusual repetitive movements?", 
                   options=["Yes", "No"], horizontal=True, key="A10")
processed_answers["A10"] = 1 if ans_a10 == "Yes" else 0

if st.button("Analyze Results"):
    input_data = pd.DataFrame([[
        processed_answers['A1'], processed_answers['A2'], processed_answers['A3'], 
        processed_answers['A4'], processed_answers['A5'], processed_answers['A6'], 
        processed_answers['A7'], processed_answers['A8'], processed_answers['A9'], 
        processed_answers['A10'],
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
