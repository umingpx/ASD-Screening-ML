import streamlit as st
import pandas as pd
import joblib
import base64
import shap
import matplotlib.pyplot as plt

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    [data-testid="stVerticalBlock"] {{
        background-color: rgba(255, 255, 255, 0.6); 
        padding: 20px;
        border-radius: 10px;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html = True)

set_background('background.png')

model = joblib.load('asd_model.pkl')

st.set_page_config(page_title="ASD Screening Tool", page_icon="🩺")

st.title("Pediatric ASD Screening Tool")
st.write("Uses a Machine Learning model to detect ASD traits based on the Q-CHAT-10 criteria.")

st.sidebar.header("Demographics")
age = st.sidebar.slider("Age (Months)", 12, 36, 24)
sex = st.sidebar.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
jaundice = st.sidebar.selectbox("Born with Jaundice?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
family = st.sidebar.selectbox("Family History of ASD?", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
who_options = ["Health Care Professional", "Family Member", "Self"]
who = st.sidebar.selectbox("Who is completing this?", options=[0, 1, 2], format_func=lambda x: who_options[x])
eth_options = ["Middle Eastern","White European","Hispanic","Black","Asian","South Asian","Native Indian","Latino","Mixed","Pacifica","Others"]
eth = st.sidebar.selectbox("Ethnicity", options=list(range(len(eth_options))), format_func=lambda x: eth_options[x])

st.header("Behavioral Observations")
processed_answers = {}
pos_milestone_questions = {
    "A1": "Does your child look at you when you call their name?",
    "A2": "Is it easy for you to get eye contact with your child?",
    "A3": "Does your child point to indicate that s/he wants something?",
    "A4": "Does your child point to share interest/enjoyment with you?",
    "A5": "Does your child pretend? (e.g., caring for a doll, talking on a toy phone?)",
    "A6": "Does your child follow where you are looking?",
    "A7": "If you are visibly upset, does your child show signs of wanting to comfort you?",
    "A9": "Does your child use simple gestures (e.g., waving goodbye)?"
}

for key, text in pos_milestone_questions.items():
    ans = st.radio(text, options=["Yes", "No"], horizontal=True, key=key)
    processed_answers[key] = 1 if ans == "No" else 0

ans_a8 = st.radio("Would you describe your child's first words as typical or atypical?", 
                  options=["Typical", "Atypical/Delayed"], horizontal=True, key="A8")
processed_answers["A8"] = 1 if ans_a8 == "Atypical/Delayed" else 0

ans_a10 = st.radio("Does your child stare at nothing with no apparent purpose or have unusual repetitive movements?", 
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

    # START OF SHAP EXPANSION
    st.subheader("Clinical Reasoning (XAI)")
    
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        if isinstance(shap_values, list):
            current_vals = shap_values[1][0]
        elif len(shap_values.shape) == 3:
            current_vals = shap_values[0, :, 1]
        else:
            current_vals = shap_values.flatten()

        clinical_map = {
            'A1': 'Social Attention',
            'A2': 'Eye Contact',
            'A3': 'Requesting (Pointing)',
            'A4': 'Joint Attention',
            'A5': 'Pretend Play',
            'A6': 'Gaze Following',
            'A7': 'Social Reciprocity',
            'A8': 'Language Development',
            'A9': 'Gestural Communication',
            'A10': 'Repetitive Behaviors',
            'Age_Mons': 'Age (Months)',
            'Sex': 'Sex',
            'Ethnicity': 'Ethnicity',
            'Jaundice': 'Jaundice History',
            'Family_with_ASD': 'Family ASD History',
            'Who completed the test': 'Test Respondent'
        }

        feature_names = input_data.columns.tolist()
        shap_df = pd.DataFrame({
            'Feature': [clinical_map.get(f, f) for f in feature_names],
            'Impact': current_vals
        }).sort_values(by='Impact', ascending=True)

        fig, ax = plt.subplots(figsize=(8, 7))
        colors = ['#faa4a0' if x > 0 else "#d2ffd2" for x in shap_df['Impact']]
        ax.barh(shap_df['Feature'], shap_df['Impact'], color=colors)
        ax.set_xlabel("Contribution toward Prediction")
        ax.set_title("Clinical Drivers of this Result")
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout()
        st.pyplot(fig)
        st.write("**XAI Analysis:** This chart identifies which specific clinical indicators drove the AI's risk assessment. **RED** represents traits associated with ASD, while **GREEN** represents typical developmental markers.")

    except Exception as e:
        st.warning("Clinical reasoning chart is unavailable for this specific profile.")
    # END OF SHAP EXPANSION

    if prediction == 1:
        st.error(f"Potential ASD Traits Detected. Confidence: {probability:.2%}")
        st.write("Result Analysis: The behavioral patterns entered align with common clinical indicators of ASD.")
    else:
        st.success(f"No ASD Traits Detected. Confidence: {1-probability:.2%}")
        st.write("Result Analysis: The behavioral patterns entered do not show a strong correlation with ASD indicators.")

st.info("Disclaimer: This is a screening tool, not a clinical diagnosis.")
