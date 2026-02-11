import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import os

# Set page configuration
st.set_page_config(page_title="Multiple Disease Prediction System",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Load the saved models
MODELS_DIR = "."

def load_model(filename):
    path = os.path.join(MODELS_DIR, filename)
    if os.path.exists(path):
        return pickle.load(open(path, 'rb'))
    else:
        return None

diabetes_model = load_model('diabetes_model.sav')
heart_disease_model = load_model('heart_disease_model.sav')
parkinsons_model = load_model('parkinsons_model.sav')

# sidebar for navigation
with st.sidebar:
    selected = option_menu('Multiple Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction'],
                           icons=['activity', 'heart', 'person'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')

    if diabetes_model is None:
        st.error("Model not found. Please train the `diabetes_model.sav` first.")
    else:
        # getting the input data from the user
        col1, col2, col3 = st.columns(3)

        with col1:
            Pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, step=1)
        with col2:
            Glucose = st.number_input('Glucose Level', min_value=0, max_value=500)
        with col3:
            BloodPressure = st.number_input('Blood Pressure value', min_value=0, max_value=200)
        with col1:
            SkinThickness = st.number_input('Skin Thickness value', min_value=0, max_value=100)
        with col2:
            Insulin = st.number_input('Insulin Level', min_value=0, max_value=1000)
        with col3:
            BMI = st.number_input('BMI value', min_value=0.0, max_value=100.0)
        with col1:
            DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=3.0)
        with col2:
            Age = st.number_input('Age of the Person', min_value=0, max_value=120)

        # code for Prediction
        diab_diagnosis = ''

        # creating a button for Prediction
        if st.button('Diabetes Test Result'):
            try:
                user_input = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,
                              BMI, DiabetesPedigreeFunction, Age]
                user_input = [float(x) for x in user_input]
                diab_prediction = diabetes_model.predict([user_input])

                if diab_prediction[0] == 1:
                    diab_diagnosis = 'The person is diabetic'
                else:
                    diab_diagnosis = 'The person is not diabetic'
                st.success(diab_diagnosis)
            except ValueError:
                st.error("Please enter valid numerical values for all fields.")

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')

    if heart_disease_model is None:
        st.error("Model not found. Please train the `heart_disease_model.sav` first.")
    else:
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input('Age', min_value=1, max_value=120, value=60)
        with col2:
            sex = st.selectbox('Sex', ['Male', 'Female'])
            sex_val = 1 if sex == 'Male' else 0
        with col3:
            cp = st.selectbox('Chest Pain types', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
            # Mapping: asymptomatic:0, atypical angina:1, non-anginal:2, typical angina:3
            if cp == 'Asymptomatic': cp_val = 0
            elif cp == 'Atypical Angina': cp_val = 1
            elif cp == 'Non-anginal Pain': cp_val = 2
            else: cp_val = 3

        with col1:
            trestbps = st.number_input('Resting Blood Pressure', min_value=50, max_value=250, value=120)
        with col2:
            chol = st.number_input('Serum Cholestoral in mg/dl', min_value=100, max_value=600, value=200)
        with col3:
            fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['True', 'False'])
            fbs_val = 1 if fbs == 'True' else 0

        with col1:
            restecg = st.selectbox('Resting Electrocardiographic results', ['Normal', 'LV Hypertrophy', 'ST-T Wave Abnormality'])
            # Mapping: lv hypertrophy:0, normal:2, st-t abnormality:3
            if restecg == 'LV Hypertrophy': restecg_val = 0
            elif restecg == 'Normal': restecg_val = 2
            else: restecg_val = 3

        with col2:
            thalach = st.number_input('Maximum Heart Rate achieved', min_value=50, max_value=250, value=150)
        with col3:
            exang = st.selectbox('Exercise Induced Angina', ['Yes', 'No'])
            exang_val = 1 if exang == 'Yes' else 0

        with col1:
            oldpeak = st.number_input('ST depression induced by exercise', min_value=0.0, max_value=10.0, value=1.0)
        with col2:
            slope = st.selectbox('Slope of the peak exercise ST segment', ['Upsloping', 'Flat', 'Downsloping'])
            # Mapping: downsloping:0, flat:1, upsloping:3
            if slope == 'Downsloping': slope_val = 0
            elif slope == 'Flat': slope_val = 1
            else: slope_val = 3

        with col3:
            ca = st.number_input('Major vessels colored by flourosopy', min_value=0, max_value=4, value=0)
        with col1:
            thal = st.selectbox('thal', ['Normal', 'Fixed Defect', 'Reversable Defect'])
            # Mapping: fixed defect:0, normal:2, reversable defect:3
            if thal == 'Fixed Defect': thal_val = 0
            elif thal == 'Normal': thal_val = 2
            else: thal_val = 3

        heart_diagnosis = ''

        if st.button('Heart Disease Test Result'):
            try:
                # user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
                user_input = [age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, thalach, exang_val, oldpeak, slope_val, ca, thal_val]
                user_input = [float(x) for x in user_input]
                heart_prediction = heart_disease_model.predict([user_input])

                if heart_prediction[0] == 1:
                    heart_diagnosis = 'The person is having heart disease'
                else:
                    heart_diagnosis = 'The person does not have any heart disease'
                st.success(heart_diagnosis)
            except ValueError:
                st.error("Please enter valid numerical values for all fields.")

# Parkinson's Prediction Page
if selected == "Parkinsons Prediction":
    st.title("Parkinson's Disease Prediction using ML")

    if parkinsons_model is None:
        st.error("Model not found. Please train the `parkinsons_model.sav` first.")
    else:
        st.info("Note: This model expects input features from the 'Parkinsons Disease Data Set' (Classification). If you are using the Telemonitoring dataset, these predictions may not be applicable.")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            fo = st.text_input('MDVP:Fo(Hz)')
        with col2:
            fhi = st.text_input('MDVP:Fhi(Hz)')
        with col3:
            flo = st.text_input('MDVP:Flo(Hz)')
        with col4:
            Jitter_percent = st.text_input('MDVP:Jitter(%)')
        with col5:
            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
        with col1:
            RAP = st.text_input('MDVP:RAP')
        with col2:
            PPQ = st.text_input('MDVP:PPQ')
        with col3:
            DDP = st.text_input('Jitter:DDP')
        with col4:
            Shimmer = st.text_input('MDVP:Shimmer')
        with col5:
            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
        with col1:
            APQ3 = st.text_input('Shimmer:APQ3')
        with col2:
            APQ5 = st.text_input('Shimmer:APQ5')
        with col3:
            APQ = st.text_input('MDVP:APQ')
        with col4:
            DDA = st.text_input('Shimmer:DDA')
        with col5:
            NHR = st.text_input('NHR')
        with col1:
            HNR = st.text_input('HNR')
        with col2:
            RPDE = st.text_input('RPDE')
        with col3:
            DFA = st.text_input('DFA')
        with col4:
            spread1 = st.text_input('spread1')
        with col5:
            spread2 = st.text_input('spread2')
        with col1:
            D2 = st.text_input('D2')
        with col2:
            PPE = st.text_input('PPE')

        parkinsons_diagnosis = ''

        if st.button("Parkinson's Test Result"):
            try:
                user_input = [fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ, DDP, Shimmer, Shimmer_dB, APQ3, APQ5,
                              APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]
                user_input = [float(x) for x in user_input]
                parkinsons_prediction = parkinsons_model.predict([user_input])

                if parkinsons_prediction[0] == 1:
                    parkinsons_diagnosis = "The person has Parkinson's disease"
                else:
                    parkinsons_diagnosis = "The person does not have Parkinson's disease"
                st.success(parkinsons_diagnosis)
            except ValueError:
                st.error("Please enter valid numerical values for all fields.")
