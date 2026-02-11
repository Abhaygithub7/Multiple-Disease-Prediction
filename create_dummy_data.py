import pandas as pd
import numpy as np
import os

DATA_DIR = "./data"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def create_diabetes_data():
    print("Creating dummy Diabetes dataset...")
    # Schema: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome
    n_samples = 200
    data = {
        'Pregnancies': np.random.randint(0, 15, n_samples),
        'Glucose': np.random.randint(50, 200, n_samples),
        'BloodPressure': np.random.randint(40, 120, n_samples),
        'SkinThickness': np.random.randint(0, 100, n_samples),
        'Insulin': np.random.randint(0, 800, n_samples),
        'BMI': np.random.uniform(15, 50, n_samples),
        'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.5, n_samples),
        'Age': np.random.randint(21, 80, n_samples),
        'Outcome': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_DIR, "diabetes.csv"), index=False)
    print("diabetes.csv created.")

def create_heart_data():
    print("Creating dummy Heart Disease dataset...")
    # Schema: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target
    n_samples = 200
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(100, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 220, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 5, n_samples),
        'thal': np.random.randint(0, 4, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_DIR, "heart.csv"), index=False)
    print("heart.csv created.")

def create_parkinsons_data():
    print("Creating dummy Parkinson's dataset...")
    # Schema: name, MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA, NHR, HNR, status, RPDE, DFA, spread1, spread2, D2, PPE
    n_samples = 200
    data = {
        'name': [f"subject_{i}" for i in range(n_samples)],
        'MDVP:Fo(Hz)': np.random.uniform(100, 250, n_samples),
        'MDVP:Fhi(Hz)': np.random.uniform(150, 600, n_samples),
        'MDVP:Flo(Hz)': np.random.uniform(50, 200, n_samples),
        'MDVP:Jitter(%)': np.random.uniform(0, 0.02, n_samples),
        'MDVP:Jitter(Abs)': np.random.uniform(0, 0.0001, n_samples),
        'MDVP:RAP': np.random.uniform(0, 0.01, n_samples),
        'MDVP:PPQ': np.random.uniform(0, 0.01, n_samples),
        'Jitter:DDP': np.random.uniform(0, 0.03, n_samples),
        'MDVP:Shimmer': np.random.uniform(0, 0.1, n_samples),
        'MDVP:Shimmer(dB)': np.random.uniform(0, 1.0, n_samples),
        'Shimmer:APQ3': np.random.uniform(0, 0.05, n_samples),
        'Shimmer:APQ5': np.random.uniform(0, 0.06, n_samples),
        'MDVP:APQ': np.random.uniform(0, 0.08, n_samples),
        'Shimmer:DDA': np.random.uniform(0, 0.15, n_samples),
        'NHR': np.random.uniform(0, 0.1, n_samples),
        'HNR': np.random.uniform(15, 30, n_samples),
        'status': np.random.randint(0, 2, n_samples),
        'RPDE': np.random.uniform(0.2, 0.7, n_samples),
        'DFA': np.random.uniform(0.5, 0.8, n_samples),
        'spread1': np.random.uniform(-7, -2, n_samples),
        'spread2': np.random.uniform(0.1, 0.5, n_samples),
        'D2': np.random.uniform(1.5, 3.5, n_samples),
        'PPE': np.random.uniform(0.1, 0.4, n_samples)
    }
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA_DIR, "parkinsons.csv"), index=False)
    print("parkinsons.csv created.")

if __name__ == "__main__":
    create_diabetes_data()
    create_heart_data()
    create_parkinsons_data()
