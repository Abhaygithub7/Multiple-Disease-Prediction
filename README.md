# Multiple Disease Prediction System 🏥

A user-friendly web application built with **Python** and **Streamlit** that predicts the likelihood of **Diabetes**, **Heart Disease**, and **Parkinson's Disease** using Machine Learning.

## 🚀 Features
-   **Diabetes Prediction**: Predicts diabetes risk based on medical metrics (Glucose, BMI, Age, etc.).
-   **Heart Disease Prediction**: assesses heart disease risk using cardiovascular parameters (Chest Pain, Cholesterol, etc.).
-   **Parkinson's Prediction**: Detects Parkinson's disease using voice frequency measurements (Jitter, Shimmer, etc.).
-   **Interactive UI**: Simple and intuitive interface powered by Streamlit.
-   **Data Visualization**: Includes scripts to generate insight charts (Confusion Matrices, Feature Importance).

## 🛠️ Tech Stack
-   **Frontend**: Streamlit
-   **Backend/ML**: Python, Scikit-Learn, Pandas, NumPy
-   **Algorithms**: Random Forest Classifier (Optimized for higher accuracy)
-   **Visualization**: Matplotlib, Seaborn

## 📊 Model Performance
We trained our models using **Random Forest Classifiers** for robust performance:
-   **Diabetes Model**: ~72% Accuracy (PIMA Dataset)
-   **Heart Disease Model**: ~84% Accuracy (UCI Heart Disease)
-   **Parkinson's Model**: ~82% Accuracy (UCI Parkinson's Disease)

## 📂 Project Structure
```
multiple_disease_prediction/
├── app.py                   # Main Streamlit Application
├── train_models.py          # Script to train and save ML models
├── visualize_models.py      # Script to generate data visualizations
├── requirements.txt         # Project dependencies
├── data/                    # Dataset folder (CSV files)
├── visualizations/          # Generated charts and graphs
└── models/                  # Saved .sav model files (generated after training)
```

## ⚙️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Abhaygithub7/Multiple-Disease-Prediction.git
    cd Multiple-Disease-Prediction
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Application**
    ```bash
    streamlit run app.py
    ```

4.  **(Optional) Retrain Models**
    If you want to retrain the models with your own data:
    ```bash
    python3 train_models.py
    ```

5.  **(Optional) Generate Visualizations**
    To see data distributions and model evaluation charts:
    ```bash
    python3 visualize_models.py
    ```

## 📝 Datasets
This project uses public datasets from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php):
-   PIMA Indians Diabetes Database
-   Heart Disease Dataset (Cleveland)
-   Parkinson's Disease Data Set (Oxford)

## 🤝 Contribution
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.

## 📜 License
This project is open-source and available under the MIT License.
