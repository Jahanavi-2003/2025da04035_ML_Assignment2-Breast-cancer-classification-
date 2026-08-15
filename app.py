import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)


# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🩺 Breast Cancer Classification")

st.write(
    """
    This application compares machine learning classification models
    for predicting whether a breast tumor is benign or malignant.
    """
)


# ---------------------------------------------------
# MODEL FILES
# ---------------------------------------------------

model_files = {
    "Logistic Regression": "model/logistic_regression.pkl",
    "Decision Tree": "model/decision_tree.pkl",
    "K-Nearest Neighbors": "model/knn.pkl",
    "Gaussian Naive Bayes": "model/naive_bayes.pkl",
    "Random Forest": "model/random_forest.pkl"
}


# ---------------------------------------------------
# MODEL SELECTION
# ---------------------------------------------------

st.sidebar.header("Model Selection")

selected_model = st.sidebar.selectbox(
    "Select a Classification Model",
    list(model_files.keys())
)


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

model_path = model_files[selected_model]

model = joblib.load(model_path)

st.sidebar.success(
    f"Selected Model: {selected_model}"
)


# ---------------------------------------------------
# DATASET UPLOAD
# ---------------------------------------------------

st.header("1. Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)


# ---------------------------------------------------
# PROCESS DATA
# ---------------------------------------------------

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.success("Dataset uploaded successfully!")

    st.subheader("Uploaded Dataset")

    st.dataframe(data)


    # ------------------------------------------------
    # CHECK TARGET COLUMN
    # ------------------------------------------------

    if "diagnosis" not in data.columns:

        st.error(
            "The uploaded CSV must contain a 'diagnosis' column."
        )

    else:

        # Separate features and target

        X = data.drop("diagnosis", axis=1)

        y = data["diagnosis"]


        # ------------------------------------------------
        # TARGET CONVERSION
        # ------------------------------------------------

        # Handle B/M format if present

        if y.dtype == "object":

            y = y.map({
                "B": 0,
                "M": 1,
                "Benign": 0,
                "Malignant": 1
            })


        # Convert to integer

        y = y.astype(int)


        # ------------------------------------------------
        # PREDICTIONS
        # ------------------------------------------------

        y_pred = model.predict(X)

        y_prob = model.predict_proba(X)[:, 1]


        # ------------------------------------------------
        # EVALUATION METRICS
        # ------------------------------------------------

        accuracy = accuracy_score(
            y,
            y_pred
        )

        auc = roc_auc_score(
            y,
            y_prob
        )

        precision = precision_score(
            y,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y,
            y_pred
        )


        # ------------------------------------------------
        # DISPLAY METRICS
        # ------------------------------------------------

        st.header("2. Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Accuracy",
                f"{accuracy:.4f}"
            )

            st.metric(
                "Precision",
                f"{precision:.4f}"
            )


        with col2:

            st.metric(
                "AUC Score",
                f"{auc:.4f}"
            )

            st.metric(
                "Recall",
                f"{recall:.4f}"
            )


        with col3:

            st.metric(
                "F1 Score",
                f"{f1:.4f}"
            )

            st.metric(
                "MCC Score",
                f"{mcc:.4f}"
            )


        # ------------------------------------------------
        # CONFUSION MATRIX
        # ------------------------------------------------

        st.header("3. Confusion Matrix")

        cm = confusion_matrix(
            y,
            y_pred
        )


        fig, ax = plt.subplots(
            figsize=(6, 5)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[
                "Benign",
                "Malignant"
            ],
            yticklabels=[
                "Benign",
                "Malignant"
            ],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        ax.set_title(
            f"Confusion Matrix - {selected_model}"
        )

        st.pyplot(fig)


        # ------------------------------------------------
        # CLASSIFICATION REPORT
        # ------------------------------------------------

        st.header("4. Classification Report")

        report = classification_report(
            y,
            y_pred,
            target_names=[
                "Benign",
                "Malignant"
            ],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.round(4)
        )


        # ------------------------------------------------
        # PREDICTION RESULTS
        # ------------------------------------------------

        st.header("5. Prediction Results")

        results = X.copy()

        results["Actual Diagnosis"] = y

        results["Predicted Diagnosis"] = y_pred

        results["Predicted Class"] = results[
            "Predicted Diagnosis"
        ].map({
            0: "Benign",
            1: "Malignant"
        })

        st.dataframe(
            results
        )


else:

    st.info(
        "Please upload the test_data.csv file to begin."
    )