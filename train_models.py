import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os

# --- Configuration ---
DATA_DIR = "./data"
MODELS_DIR = "."

def train_diabetes_model():
    print("Training Diabetes Model...")
    data_path = os.path.join(DATA_DIR, "diabetes.csv")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Skipping Diabetes model training.")
        return

    # Check if file has header or requires one
    df_preview = pd.read_csv(data_path, nrows=1)
    if 'Glucose' in df_preview.columns:
        diabetes_dataset = pd.read_csv(data_path)
    else:
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        diabetes_dataset = pd.read_csv(data_path, names=columns, header=None)

    # Basic cleanup
    for col in diabetes_dataset.columns:
        diabetes_dataset[col] = pd.to_numeric(diabetes_dataset[col], errors='coerce')
    diabetes_dataset.dropna(inplace=True)

    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # OPTIMIZATION: Using Random Forest instead of SVM
    classifier = RandomForestClassifier(n_estimators=100, random_state=2)
    classifier.fit(X_train, Y_train)

    training_accuracy = accuracy_score(classifier.predict(X_train), Y_train)
    test_accuracy = accuracy_score(classifier.predict(X_test), Y_test)
    print(f"Diabetes Model (Random Forest) Accuracy - Training: {training_accuracy:.2f}, Test: {test_accuracy:.2f}")

    filename = os.path.join(MODELS_DIR, 'diabetes_model.sav')
    pickle.dump(classifier, open(filename, 'wb'))
    print(f"Diabetes model saved to {filename}")

def train_heart_disease_model():
    print("Training Heart Disease Model...")
    data_path = os.path.join(DATA_DIR, "heart.csv")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Skipping Heart Disease model training.")
        return

    heart_data = pd.read_csv(data_path)
    
    # --- Preprocessing for Real World Data (Cleveland/UCI) ---
    if 'num' in heart_data.columns:
        heart_data.rename(columns={'num': 'target'}, inplace=True)
    elif 'target' not in heart_data.columns:
        heart_data.rename(columns={heart_data.columns[-1]: 'target'}, inplace=True)

    heart_data['target'] = pd.to_numeric(heart_data['target'], errors='coerce').fillna(0).astype(int)
    heart_data['target'] = heart_data['target'].apply(lambda x: 1 if x > 0 else 0)

    if 'thalch' in heart_data.columns:
        heart_data.rename(columns={'thalch': 'thalach'}, inplace=True)
        
    for col in ['id', 'dataset']:
        if col in heart_data.columns:
            heart_data.drop(columns=[col], inplace=True)

    cat_columns = heart_data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for col in cat_columns:
        heart_data[col] = heart_data[col].astype(str)
        heart_data[col] = le.fit_transform(heart_data[col])
        print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    for col in heart_data.columns:
        heart_data[col] = pd.to_numeric(heart_data[col], errors='coerce')

    original_len = len(heart_data)
    heart_data.dropna(inplace=True)
    if len(heart_data) < original_len:
        print(f"Dropped {original_len - len(heart_data)} rows with missing values.")

    expected_cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    available_cols = [c for c in expected_cols if c in heart_data.columns]
    
    X = heart_data[available_cols]
    Y = heart_data['target']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # OPTIMIZATION: Using Random Forest instead of Logistic Regression
    model = RandomForestClassifier(n_estimators=100, random_state=2)
    model.fit(X_train, Y_train)

    training_accuracy = accuracy_score(model.predict(X_train), Y_train)
    test_accuracy = accuracy_score(model.predict(X_test), Y_test)
    print(f"Heart Disease Model (Random Forest) Accuracy - Training: {training_accuracy:.2f}, Test: {test_accuracy:.2f}")

    filename = os.path.join(MODELS_DIR, 'heart_disease_model.sav')
    pickle.dump(model, open(filename, 'wb'))
    print(f"Heart Disease model saved to {filename}")

def train_parkinsons_model():
    print("Training Parkinson's Disease Model...")
    data_path = os.path.join(DATA_DIR, "parkinsons.csv")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}. Skipping Parkinson's model training.")
        return

    parkinsons_data = pd.read_csv(data_path)
    
    # CHECK FOR COMPATIBILITY
    if 'motor_UPDRS' in parkinsons_data.columns:
        print("WARNING: 'Parkinsons Telemonitoring' dataset detected (Regression).")
        print("The current App requires the 'Classification' dataset (Binary: Healthy vs Parkinson's).")
        print("This dataset lacks healthy controls and targets 'UPDRS' score, not disease status.")
        print("SKIPPING Parkinson's training to avoid incorrect model generation.")
        print("Please upload the UCI 'Parkinsons Data Set' (containing 'MDVP:Fo(Hz)' and 'status' column).")
        return

    if 'name' in parkinsons_data.columns:
        X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
        Y = parkinsons_data['status']
    elif 'status' in parkinsons_data.columns:
         X = parkinsons_data.drop(columns=['status'], axis=1)
         Y = parkinsons_data['status']
    else:
         print("Could not find 'status' column in Parkinson's data. Skipping.")
         return

    # scaler = StandardScaler()
    # scaler.fit(X)
    # X = scaler.transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # OPTIMIZATION: Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=2)
    model.fit(X_train, Y_train)

    training_accuracy = accuracy_score(model.predict(X_train), Y_train)
    test_accuracy = accuracy_score(model.predict(X_test), Y_test)
    print(f"Parkinson's Model (Random Forest) Accuracy - Training: {training_accuracy:.2f}, Test: {test_accuracy:.2f}")

    filename = os.path.join(MODELS_DIR, 'parkinsons_model.sav')
    pickle.dump(model, open(filename, 'wb'))
    print(f"Parkinson's model saved to {filename}")

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    if not os.path.exists(DATA_DIR):
        try:
            os.makedirs(DATA_DIR)
            print(f"Created directory: {DATA_DIR}. Please place your datasets (diabetes.csv, heart.csv, parkinsons.csv) here.")
        except OSError as e:
            print(f"Error creating directory {DATA_DIR}: {e}")
    
    train_diabetes_model()
    print("-" * 20)
    train_heart_disease_model()
    print("-" * 20)
    train_parkinsons_model()
