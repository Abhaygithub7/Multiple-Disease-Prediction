# Machine Learning Model Source Code

This document contains the Python source code used to train the machine learning models for the **Multiple Disease Prediction System**. These snippets demonstrate the data preprocessing, model selection (Random Forest), and evaluation logic.

## 1. Diabetes Prediction Model

**Algorithm**: Random Forest Classifier
**Dataset**: PIMA Indians Diabetes Database

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_diabetes_model():
    # Load Dataset
    diabetes_dataset = pd.read_csv("data/diabetes.csv")
    
    # Preprocessing
    # Ensure numeric types
    for col in diabetes_dataset.columns:
        diabetes_dataset[col] = pd.to_numeric(diabetes_dataset[col], errors='coerce')
    diabetes_dataset.dropna(inplace=True)

    # Features (X) and Target (Y)
    X = diabetes_dataset.drop(columns='Outcome', axis=1)
    Y = diabetes_dataset['Outcome']

    # Standardization
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Train-Test Split (80-20)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Model Training (Random Forest)
    classifier = RandomForestClassifier(n_estimators=100, random_state=2)
    classifier.fit(X_train, Y_train)

    # Evaluation
    training_accuracy = accuracy_score(classifier.predict(X_train), Y_train)
    test_accuracy = accuracy_score(classifier.predict(X_test), Y_test)
    print(f"Diabetes Model Accuracy - Train: {training_accuracy:.2f}, Test: {test_accuracy:.2f}")

    # Save Model
    pickle.dump(classifier, open('diabetes_model.sav', 'wb'))
```

## 2. Heart Disease Prediction Model

**Algorithm**: Random Forest Classifier
**Dataset**: Heart Disease UCI (Cleveland)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_heart_disease_model():
    # Load Dataset
    heart_data = pd.read_csv("data/heart.csv")

    # Preprocessing: Handle Missing/Categorical Data
    # (Simplified for clarity)
    heart_data['target'] = pd.to_numeric(heart_data['target'], errors='coerce').fillna(0).astype(int)
    heart_data['target'] = heart_data['target'].apply(lambda x: 1 if x > 0 else 0) # Binary Classification
    
    # Label Encoding for Categorical Columns (Sex, Chest Pain, etc.)
    cat_columns = heart_data.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_columns:
        heart_data[col] = le.fit_transform(heart_data[col].astype(str))
    
    # Feature Selection
    X = heart_data.drop(columns=['target'], axis=1)
    Y = heart_data['target']

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    # Model Training (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=2)
    model.fit(X_train, Y_train)

    # Evaluation
    training_accuracy = accuracy_score(model.predict(X_train), Y_train)
    test_accuracy = accuracy_score(model.predict(X_test), Y_test)
    print(f"Heart Disease Accuracy - Train: {training_accuracy:.2f}, Test: {test_accuracy:.2f}")

    # Save Model
    pickle.dump(model, open('heart_disease_model.sav', 'wb'))
```

## 3. Parkinson's Prediction Model

**Algorithm**: Random Forest Classifier
**Dataset**: Parkinson's Disease Data Set (UCI - Classification)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

def train_parkinsons_model():
    # Load Dataset
    parkinsons_data = pd.read_csv("data/parkinsons.csv")
    
    # Preprocessing
    # Drop 'name' column as it's not a feature
    if 'name' in parkinsons_data.columns:
        X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
        Y = parkinsons_data['status']
    else:
        X = parkinsons_data.drop(columns=['status'], axis=1)
        Y = parkinsons_data['status']

    # Standardization
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

    # Model Training (Random Forest)
    model = RandomForestClassifier(n_estimators=100, random_state=2)
    model.fit(X_train, Y_train)

    # Evaluation
    training_accuracy = accuracy_score(model.predict(X_train), Y_train)
    test_accuracy = accuracy_score(model.predict(X_test), Y_test)
    print(f"Parkinsons Accuracy - Train: {training_accuracy:.2f}, Test: {test_accuracy:.2f}")

    # Save Model
    pickle.dump(model, open('parkinsons_model.sav', 'wb'))
```
