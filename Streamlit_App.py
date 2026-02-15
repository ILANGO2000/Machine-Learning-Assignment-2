# Core UI framework
import streamlit as st

# Data handling
import pandas as pd
import numpy as np

# Visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt

# Machine learning utilities
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Custom model builder
from model.models import get_model


# Page configuration
st.set_page_config(page_title="Breast Cancer Diagnosis Dashboard", layout="wide")

st.header("Machine Learning Classification Studio")

uploaded_csv = st.file_uploader("Upload your dataset in CSV format", type=["csv"])

if uploaded_csv is not None:

    data_table = pd.read_csv(uploaded_csv)
    data_table.columns = data_table.columns.str.strip()

    # Limit size for performance
    if len(data_table) > 20000:
        data_table = data_table.sample(20000, random_state=42)

    st.markdown("### Dataset Sample")
    st.dataframe(data_table.head(), use_container_width=True)

    target_field = st.selectbox("Choose the target variable", data_table.columns)

    if target_field:

        features = data_table.drop(columns=[target_field])
        target = data_table[target_field]

        # Encode labels if categorical
        if target.dtype == "object":
            encoder = LabelEncoder()
            target = encoder.fit_transform(target)

        model_choice = st.selectbox(
            "Choose a classification algorithm",
            [
                "Logistic Regression",
                "Decision Tree",
                "KNN",
                "Naive Bayes",
                "Random Forest",
                "XGBoost"
            ]
        )

        split_ratio = st.slider("Select test data proportion", 0.1, 0.5, 0.2)

        X_train, X_test, y_train, y_test = train_test_split(
            features,
            target,
            test_size=split_ratio,
            random_state=42
        )

        # -------- MODEL REFRESH STEP --------
        if st.button("Refresh Model"):

            st.session_state["current_model"] = get_model(model_choice, X_train)
            st.success("Model refreshed successfully")

        # -------- APPLY MODEL BUTTON --------
        if "current_model" in st.session_state:

            if st.button("Apply Model"):

                with st.spinner("Model is running. Please wait..."):

                    active_model = st.session_state["current_model"]

                    active_model.fit(X_train, y_train)
                    predictions = active_model.predict(X_test)

                    # Probability if supported
                    if hasattr(active_model, "predict_proba"):
                        probabilities = active_model.predict_proba(X_test)[:, 1]
                        auc_value = roc_auc_score(y_test, probabilities)
                    else:
                        auc_value = None

                    # -------- PERFORMANCE METRICS --------
                    acc = accuracy_score(y_test, predictions)
                    prec = precision_score(y_test, predictions, average="weighted")
                    rec = recall_score(y_test, predictions, average="weighted")
                    f1_val = f1_score(y_test, predictions, average="weighted")
                    mcc_val = matthews_corrcoef(y_test, predictions)

                    cv_results = cross_val_score(
                        get_model(model_choice, X_train),
                        features,
                        target,
                        cv=3,
                        n_jobs=-1
                    )

                st.markdown("## Model Evaluation")

                m1, m2, m3, m4, m5, m6 = st.columns(6)

                m1.metric("Accuracy", f"{acc:.4f}")
                m2.metric("Precision", f"{prec:.4f}")
                m3.metric("Recall", f"{rec:.4f}")
                m4.metric("F1 Score", f"{f1_val:.4f}")
                m5.metric("MCC", f"{mcc_val:.4f}")
                m6.metric("CV Score", f"{cv_results.mean():.4f}")

                if auc_value is not None:
                    st.metric("ROC AUC", f"{auc_value:.4f}")

                # -------- CONFUSION MATRIX --------
                st.markdown("## Confusion Matrix")

                matrix = confusion_matrix(y_test, predictions)

                fig, ax = plt.subplots()

                sns.heatmap(
                    matrix,
                    annot=True,
                    fmt="d",
                    cmap="coolwarm",
                    linewidths=1,
                    linecolor="black"
                )

                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("Actual Label")

                st.pyplot(fig)

                # -------- CLASSIFICATION REPORT --------
                st.markdown("## Detailed Classification Report")

                report_dict = classification_report(
                    y_test,
                    predictions,
                    output_dict=True
                )

                report_frame = pd.DataFrame(report_dict).transpose()

                st.dataframe(report_frame.round(4), use_container_width=True)