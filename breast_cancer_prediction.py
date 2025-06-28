import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

model = joblib.load("breast_cancer_model.pkl")

st.title("🎗️ Breast Cancer Recurrence Predictor")

st.markdown("""
Welcome to the **Breast Cancer Recurrence Predictor** app!  
This interactive tool uses a **Logistic Regression model** trained on a real-world breast cancer dataset to estimate the **likelihood of cancer recurrence** based on patient features.

---

### Input Features Explained

You can adjust the following features to simulate different patient profiles:

- **Age Range**: Grouped age categories from 30 to 79 years  
- **Menopause Status**: Whether the patient is premenopausal or had menopause before age 40  
- **Tumor Size**: Ranges like "0-4", "5-9", ..., "50-54" in millimeters  
- **Invasive Nodes (Lymph Nodes)**: Count ranges of nodes with signs of cancer (e.g., "0-2", "3-5")  
- **Node Caps**: Whether cancer has spread beyond the lymph node capsule  
- **Breast Side**: Which breast was affected — left or right  
- **Tumor Quadrant**: The quadrant of the breast where the tumor was located  
- **Radiation Therapy**: Whether the patient received radiation therapy  
- **Degree of Malignancy**: Rated from 1 (low) to 3 (high)

---

""")


age = st.selectbox("Age Range", ["30-39", "40-49", "50-59", "60-69", "70-79"])
menopause = st.selectbox("Menopause", ["lt40", "premeno"])
tumor_size = st.selectbox("Tumor Size", ["05-09", "10-14", "15-19", "20-24", "25-29", "30-34", "35-39", "40-44", "45-49", "50-54"])
inv_nodes = st.selectbox("Invasive Nodes Range", ["03-05", "06-08", "09-11", "12-14", "15-17", "24-26"])
node_caps = st.radio("Node Caps Present?", ["yes", "no"])
breast = st.radio("Breast Side", ["left", "right"])
quad = st.selectbox("Tumor Location in Breast Quadrant", ["left_low", "left_up", "right_low", "right_up"])
irradiat = st.radio("Received Radiation Therapy?", ["yes", "no"])
deg_malig = st.selectbox("Degree of Malignancy", [1, 2, 3])






input_data = {
    "deg-malig": deg_malig,

    # Age
    "age_30-39": 1 if age == "30-39" else 0,
    "age_40-49": 1 if age == "40-49" else 0,
    "age_50-59": 1 if age == "50-59" else 0,
    "age_60-69": 1 if age == "60-69" else 0,
    "age_70-79": 1 if age == "70-79" else 0,

    # Menopause
    "menopause_lt40": 1 if menopause == "lt40" else 0,
    "menopause_premeno": 1 if menopause == "premeno" else 0,

    # Tumor size
    "tumor-size_09-May": 1 if tumor_size == "05-09" else 0,
    "tumor-size_14-Oct": 1 if tumor_size == "10-14" else 0,
    "tumor-size_15-19": 1 if tumor_size == "15-19" else 0,
    "tumor-size_20-24": 1 if tumor_size == "20-24" else 0,
    "tumor-size_25-29": 1 if tumor_size == "25-29" else 0,
    "tumor-size_30-34": 1 if tumor_size == "30-34" else 0,
    "tumor-size_35-39": 1 if tumor_size == "35-39" else 0,
    "tumor-size_40-44": 1 if tumor_size == "40-44" else 0,
    "tumor-size_45-49": 1 if tumor_size == "45-49" else 0,
    "tumor-size_50-54": 1 if tumor_size == "50-54" else 0,

    # Inv nodes
    "inv-nodes_05-Mar": 1 if inv_nodes == "03-05" else 0,
    "inv-nodes_08-Jun": 1 if inv_nodes == "06-08" else 0,
    "inv-nodes_11-Sep": 1 if inv_nodes == "09-11" else 0,
    "inv-nodes_14-Dec": 1 if inv_nodes == "12-14" else 0,
    "inv-nodes_15-17": 1 if inv_nodes == "15-17" else 0,
    "inv-nodes_24-26": 1 if inv_nodes == "24-26" else 0,

    # Node caps
    "node-caps_yes": 1 if node_caps == "yes" else 0,

    # Breast side
    "breast_right": 1 if breast == "right" else 0,

    # Breast quadrant
    "breast-quad_left_low": 1 if quad == "left_low" else 0,
    "breast-quad_left_up": 1 if quad == "left_up" else 0,
    "breast-quad_right_low": 1 if quad == "right_low" else 0,
    "breast-quad_right_up": 1 if quad == "right_up" else 0,

    # Radiation
    "irradiat_yes": 1 if irradiat == "yes" else 0
}


input_df = pd.DataFrame([input_data])


if st.button("Predict Recurrence"):
    
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.success(f"Prediction: {'Recurrence' if prediction == 'recurrence-events' else 'No Recurrence'}")
    
    fig, ax = plt.subplots(figsize=(5, 1.2))
    ax.barh([''], [proba * 100], color='red' if proba > 0.5 else 'green')
    ax.set_xlim(0, 100)
    ax.set_title(f"Predicted Risk of Recurrence {round(proba * 100, 2)} %")
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    st.pyplot(fig)