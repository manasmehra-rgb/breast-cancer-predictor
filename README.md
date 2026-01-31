# 🎗️ Breast Cancer Recurrence Predictor
An ML app for predicting breast cancer recurrence risk using Streamlit.

🚀 Try out the live app [here](https://mybreastcancerpredictor.streamlit.app/)

----

<p align="center">
  <img src="https://github.com/user-attachments/assets/05d4f30d-e8de-4b98-b889-068d99857a74" alt="Breast Cancer Banner" width="80%">
</p>


----

##  Project Motivation

Breast cancer is one of the most common cancers affecting women worldwide. While treatment advancements have improved outcomes, **recurrence** remains a significant concern for survivors and doctors alike.

This project aims to build a simple, interactive tool to estimate the **risk of breast cancer recurrence** using machine learning. The aim of this project is not to replace medical advice, but to demonstrate how data science can aid healthcare insights.

---

##  The Journey: From Raw Data to an Interactive App


### 🔹 1. Data Cleaning & Preprocessing
- Started with a real-world breast cancer dataset with categorical features.
- Applied **one-hot encoding** to convert string categories into numeric values.
- Cleaned inconsistencies and ensured the data was model-ready.

### 🔹 2. Handling Class Imbalance
- The dataset was **imbalanced**, with far fewer recurrence cases.
- Applied **Undersampling**(*Random Undersampler*), and **Oversampling**( *SMOTE* & *SMOTE Tomek Links* ) to balance the dataset.
- Evaluated performance pre- and post-balancing with classification reports.
- Finally decided to go with *SMOTE Tomek Links* as this had the best model performance

### 🔹 3. Model Building
- Trained a **Logistic Regression model** on the processed data.
- Evaluated it using **confusion matrix**, **precision**, **recall**, and **F1-score**.
- Saved the final model using `joblib` for reuse in the web app.

### 🔹 4. Visualization
- Used **Matplotlib** and **Seaborn** to create clean visualizations:
  - Confusion matrix as a **heatmap**
  - **Bar chart** to visualize Class imbalance in the dataset
- Made visuals accessible even to non-technical viewers.

### 🔹 5. Streamlit Web App
- Built a fully interactive app using **Streamlit**:
  - Users can select patient features via dropdowns/radio buttons.
  - The app predicts the **chance of recurrence** and displays:
    - A textual result (Recurrence / No Recurrence)
    - Estimated probability
    - A simple bar visual showing predicted risk
- Easy for **non-technical stakeholders** to explore "what-if" scenarios.

---

