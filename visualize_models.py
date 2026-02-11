import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import os

# Configuration
DATA_DIR = "./data"
VIS_DIR = "./visualizations"

if not os.path.exists(VIS_DIR):
    os.makedirs(VIS_DIR)

# Set global style
sns.set_theme(style="whitegrid")

def plot_confusion_matrix(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(VIS_DIR, filename))
    plt.close()
    print(f"Saved {filename}")

def plot_feature_importance(model, feature_names, title, filename):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, filename))
    plt.close()
    print(f"Saved {filename}")

def visualize_diabetes():
    print("--- Visualizing Diabetes Data ---")
    data_path = os.path.join(DATA_DIR, "diabetes.csv")
    if not os.path.exists(data_path):
        print("Diabetes data not found.")
        return

    # Load Data
    df_preview = pd.read_csv(data_path, nrows=1)
    if 'Glucose' in df_preview.columns:
        df = pd.read_csv(data_path)
    else:
        columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        df = pd.read_csv(data_path, names=columns, header=None)

    # 1. Distribution Plot (Before)
    plt.figure(figsize=(12, 10))
    df.hist(figsize=(12, 10))
    plt.suptitle("Diabetes Dataset - Feature Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "diabetes_distributions.png"))
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Diabetes Correlation Matrix")
    plt.savefig(os.path.join(VIS_DIR, "diabetes_correlation.png"))
    plt.close()

    # Training for Evaluation Plots (After)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)
    
    X = df.drop(columns='Outcome')
    Y = df['Outcome']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    model = RandomForestClassifier(n_estimators=100, random_state=2)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    # 3. Confusion Matrix
    plot_confusion_matrix(Y_test, y_pred, "Diabetes Confusion Matrix", "diabetes_confusion_matrix.png")

    # 4. Feature Importance
    plot_feature_importance(model, X.columns, "Diabetes Feature Importance", "diabetes_feature_importance.png")

def visualize_heart():
    print("--- Visualizing Heart Disease Data ---")
    data_path = os.path.join(DATA_DIR, "heart.csv")
    if not os.path.exists(data_path):
        print("Heart data not found.")
        return

    df = pd.read_csv(data_path)
    
    # Minimal Preprocessing for Plotting
    if 'target' not in df.columns and 'num' in df.columns:
        df.rename(columns={'num': 'target'}, inplace=True)
    elif 'target' not in df.columns:
        df.rename(columns={df.columns[-1]: 'target'}, inplace=True)
        
    df['target'] = pd.to_numeric(df['target'], errors='coerce').fillna(0).astype(int)
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Clean for numerical correlation
    df_numeric = df.select_dtypes(include=[np.number])

    # 1. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Heart Disease Correlation Matrix")
    plt.savefig(os.path.join(VIS_DIR, "heart_correlation.png"))
    plt.close()

    # Train for Evaluation
    # ... (Simplified training logic primarily for feature importance) ...
    # Drop IDs
    for col in ['id', 'dataset']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            
    # Encode categorical
    le = LabelEncoder()
    cat_columns = df.select_dtypes(include=['object']).columns
    for col in cat_columns:
        df[col] = le.fit_transform(df[col].astype(str))
        
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    X = df.drop(columns=['target'])
    Y = df['target']
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

    model = RandomForestClassifier(n_estimators=100, random_state=2)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    # 2. Confusion Matrix
    plot_confusion_matrix(Y_test, y_pred, "Heart Disease Confusion Matrix", "heart_confusion_matrix.png")

    # 3. Feature Importance
    plot_feature_importance(model, X.columns, "Heart Disease Feature Importance", "heart_feature_importance.png")

def visualize_parkinsons():
    print("--- Visualizing Parkinson's Data ---")
    data_path = os.path.join(DATA_DIR, "parkinsons.csv")
    if not os.path.exists(data_path):
        print("Parkinson's data not found.")
        return

    df = pd.read_csv(data_path)
    
    # Preprocessing
    if 'name' in df.columns:
        df.drop(columns=['name'], inplace=True)
        
    # 1. Distribution Plot (Subset)
    plt.figure(figsize=(12, 10))
    # Plotting only first 9 features to avoid clutter
    df.iloc[:, :9].hist(figsize=(12, 10))
    plt.suptitle("Parkinson's Dataset - Feature Distributions (Subset)")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "parkinsons_distributions.png"))
    plt.close()

    # 2. Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm') # Annotations off for readability due to many features
    plt.title("Parkinson's Correlation Matrix")
    plt.savefig(os.path.join(VIS_DIR, "parkinsons_correlation.png"))
    plt.close()

    # Model Evaluation
    X = df.drop(columns='status')
    Y = df['status']
    
    # No scaling for visualization consistency with training script (Random Forest)
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
    
    model = RandomForestClassifier(n_estimators=100, random_state=2)
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)

    # 3. Confusion Matrix
    plot_confusion_matrix(Y_test, y_pred, "Parkinson's Confusion Matrix", "parkinsons_confusion_matrix.png")

    # 4. Feature Importance
    plot_feature_importance(model, X.columns, "Parkinson's Feature Importance", "parkinsons_feature_importance.png")

if __name__ == "__main__":
    visualize_diabetes()
    visualize_heart()
    visualize_parkinsons()
    print("--- Visualization Complete ---")
    print(f"Charts saved to {os.path.abspath(VIS_DIR)}")
