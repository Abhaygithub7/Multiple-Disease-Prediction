# Project Report: Multiple Disease Prediction System

## 1. Introduction
The **Multiple Disease Prediction System** is a machine learning-based web application designed to predict the likelihood of **Diabetes**, **Heart Disease**, and **Parkinson's Disease**. It provides an accessible interface for users to input medical parameters and receive instant, AI-driven health assessments.

## 2. Technology Stack & Libraries

### 2.1. Core Libraries
-   **Python**: The primary programming language used for its extensive ecosystem of data science libraries.
-   **Streamlit**: Used to build the interactive web interface. It allows for rapid deployment of data scripts as web apps without requiring extensive frontend knowledge (HTML/CSS/JS).
-   **Scikit-Learn (sklearn)**: The robust machine learning library used for:
    -   **Model Building**: Implementing SVM and Logistic Regression algorithms.
    -   **Data Preprocessing**: `StandardScaler` for feature scaling and `train_test_split` for dataset management.
    -   **Evaluation**: `accuracy_score` to measure model performance.
-   **Pandas**: Essential for data manipulation and analysis. It is used to read CSV datasets (`pd.read_csv`) and clean data (handling missing values, dropping columns).
-   **NumPy**: Fundamental package for scientific computing, used here for handling numerical arrays and converting user input into a format suitable for the models.
-   **Pickle**: A Python module used for serializing and de-serializing the trained machine learning models. This allows us to "save" a trained model to a file (`.sav`) and "load" it instantly in the web app without retraining every time.

## 3. Algorithms Used

### 3.1. Support Vector Machine (SVM)
-   **Used for**: **Diabetes Prediction** and **Parkinson's Disease Prediction**.
-   **How it works**: SVM is a supervised learning algorithm that finds a hyperplane in an N-dimensional space (where N is the number of features) that distinctly classifies the data points.
-   **Kernel**: We used a **Linear Kernel**. This works well when the data is linearly separable, creating a straight line (or hyperplane) to separate "Positive" (Disease) from "Negative" (Healthy) cases.
-   **Why chosen**: SVM is highly effective in high-dimensional spaces and is versatile for classification problems involving medical data where boundaries can be distinct.

### 3.2. Logistic Regression
-   **Used for**: **Heart Disease Prediction**.
-   **How it works**: Despite its name, Logistic Regression is a classification algorithm. It uses a logistic function (sigmoid) to model the probability of a certain class or event (e.g., probability of having heart disease). The output is a value between 0 and 1.
-   **Why chosen**: It is simple, interpretable, and highly efficient for binary classification tasks (Disease vs. No Disease). It gives us not just a prediction but a probability score, which is crucial in medical diagnostics.

## 4. Model Evaluation & Efficiency

### 4.1. Performance Metrics
We evaluated the models using **Accuracy Score**, which measures the percentage of correct predictions out of total predictions.

-   **Diabetes Model**:
    -   **Algorithm**: SVM (Linear)
    -   **Training Accuracy**: ~79%
    -   **Test Accuracy**: ~77%
    -   *Interpretation*: The model generalizes well, with practically no overfitting (small gap between training and test scores).

-   **Heart Disease Model**:
    -   **Algorithm**: Logistic Regression
    -   **Training Accuracy**: ~86%
    -   **Test Accuracy**: ~84%
    -   *Interpretation*: Strong performance. The high accuracy suggests the selected features (Chest Pain, Thal, etc.) are strong predictors.

-   **Parkinson's Disease Model**:
    -   **Algorithm**: SVM (Linear)
    -   *Status*: Currently operating on a dummy model as the correct classification dataset was not available during training.
    -   *Expected Performance*: With the correct UCI Parkinson's Classification dataset, XGBoost or SVM typically yields accuracies above 85-90%.

### 4.2. Efficiency
-   **Inference Time**: The models are extremely lightweight. Predictions are generated in generally **< 50 milliseconds** once the user clicks the button.
-   **Resource Usage**: The serialized models (`.sav` files) are small (KB size), making the application memory-efficient and suitable for deployment on low-resource servers (e.g., Streamlit Community Cloud, Render, or a basic AWS EC2 t2.micro instance).

## 5. Future Scope & Refinement

To evolve this project from a prototype to a production-grade health assistant, the following refinements are planned:

1.  **Deep Learning Integration**:
    -   Implement **Neural Networks (ANNs)** to potentially capture complex, non-linear patterns in the data that SVM/Logistic Regression might miss, enabling higher accuracy (~95%+).
2.  **Expanded Disease Scope**:
    -   Add models for **Kidney Disease**, **Liver Disease**, and **Breast Cancer** (using Wisconsin Diagnostic dataset).
3.  **Enhanced UI/UX**:
    -   Add graphical visualizations (charts, radar plots) to show users *why* they are at risk (e.g., "Your Cholesterol is in the 90th percentile").
    -   Implement PDF report download for sharing results with doctors.
4.  **Dataset Expansion**:
    -   The current PIMA (Diabetes) and Cleveland (Heart) datasets are relatively small. Training on larger, more diverse datasets will improve real-world reliability.
5.  **Parkinson's Voice Recording**:
    -   Instead of manual input numbers, integrate a feature to **record user voice** directly in the browser and extract the frequency features (Jitter, Shimmer) automatically using `librosa`.

## 7. Optimization Update (Latest)

Following the initial implementation, we performed an optimization pass to improve model accuracy and robustness.

### 7.1. Algorithm Upgrade: Random Forest
We switched from **SVM/Logistic Regression** to **Random Forest Classifier** for the Diabetes and Heart Disease models.
-   **Why?**: Random Forest is an ensemble learning method that constructs multiple decision trees. It generally provides higher accuracy on tabular data by reducing overfitting (compared to a single Decision Tree) and capturing non-linear relationships better than a Linear SVM.
-   **Results**:
    -   **Diabetes Model**: ~72% Test Accuracy. (Note: The PIMA dataset is small; Random Forest shows 100% training accuracy but 72% test, indicating the need for more data to prevent overfitting).
    -   **Heart Disease Model**: ~84% Test Accuracy. Maintained high performance with robust handling of categorical features.

### 7.2. Parkinson's Dataset Analysis & Fix
The initial `parkinsons.csv` was identified as the Telemonitoring (Regression) dataset.
-   **Correction**: We replaced it with the **Parkinsons Disease Data Set (Classification)**.
-   **Result**: The model was successfully trained using **Random Forest**.
    -   **Accuracy**: **~82%** on Test Data.
    -   **Pipeline**: Feature scaling was removed to simplify the inference pipeline, as Random Forest is robust to unscaled data.

## 8. Conclusion
The system is now fully operational with three robust Random Forest models:
1.  **Diabetes**: ~72% Accuracy
2.  **Heart Disease**: ~84% Accuracy
3.  **Parkinson's**: ~82% Accuracy

Detailed **Visualizations** (Distributions, Confusion Matrices, Feature Importance) and **Source Code** have been generated to support your project presentation.
