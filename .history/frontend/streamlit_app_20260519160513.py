import streamlit as st
import requests
import pandas as pd
import os

# =====================================================
# Streamlit Config
# =====================================================

st.set_page_config(
    page_title="AI Governance System",
    layout="wide"
)

# =====================================================
# Title
# =====================================================

st.title("AI Governance System")

st.write(
    "Upload MLflow models and run governance analysis."
)

# =====================================================
# Backend URL
# =====================================================

BACKEND_URL = "http://127.0.0.1:8000"

# =====================================================
# Upload Model
# =====================================================

st.header("1. Upload MLflow Model")

uploaded_model = st.file_uploader(
    "Upload ZIP model",
    type=["zip"],
    key="model_upload"
)

if uploaded_model is not None:

    if st.button(
        "Upload Model",
        key="upload_model_button"
    ):

        files = {
            "file": (
                uploaded_model.name,
                uploaded_model,
                "application/zip"
            )
        }

        response = requests.post(
            f"{BACKEND_URL}/api/upload-model",
            files=files
        )

        if response.status_code == 200:

            result = response.json()

            st.success(
                "Model uploaded successfully"
            )

            st.session_state[
                "model_path"
            ] = result["extracted_path"]

            if "model_inspection" in result:

                st.session_state[
                    "model_features"
                ] = result[
                    "model_inspection"
                ].get(
                    "feature_names",
                    []
                )

            else:

                st.session_state[
                    "model_features"
                ] = []

        else:

            st.session_state[
                "model_features"
            ] = []

            st.error(
                response.text
            )

# =====================================================
# Dataset Configuration
# =====================================================

st.header("2. Dataset Configuration")

generate_synthetic = st.checkbox(
    "Generate Synthetic Dataset Automatically",
    key="generate_synthetic_checkbox"
)

dataset_path = None
df = None
uploaded_dataset = None

# =====================================================
# Real Dataset Upload
# =====================================================

if not generate_synthetic:

    uploaded_dataset = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"],
        key="dataset_upload"
    )

    if uploaded_dataset is not None:

        dataset_save_path = os.path.join(
            os.getcwd(),
            uploaded_dataset.name
        )

        with open(
            dataset_save_path,
            "wb"
        ) as f:

            f.write(
                uploaded_dataset.getbuffer()
            )

        dataset_path = dataset_save_path

        st.success(
            f"Dataset saved: {dataset_path}"
        )

        df = pd.read_csv(dataset_path)

        st.subheader("Dataset Preview")

        st.dataframe(df.head(10))

# =====================================================
# Synthetic Dataset Config
# =====================================================

else:

    synthetic_rows = st.number_input(
        "Number of Synthetic Rows",
        min_value=100,
        max_value=100000,
        value=1000,
        key="synthetic_rows_input"
    )

    st.info(
        "Dataset will be generated automatically "
        "from model features."
    )

# =====================================================
# Governance Configuration
# =====================================================

st.header("3. Governance Configuration")

# =====================================================
# Target Column
# =====================================================

target_column = None

# ---------------------------------------------
# Real Dataset Mode
# ---------------------------------------------

if not generate_synthetic:

    target_column = st.text_input(
        "Target Column",
        key="target_column_input"
    )

# ---------------------------------------------
# Synthetic Dataset Mode
# ---------------------------------------------

else:

    st.info(
        "Target column will be generated "
        "automatically using model predictions."
    )

    target_column = "prediction"

# =====================================================
# Sensitive Columns
# =====================================================

sensitive_columns = []

# ---------------------------------------------
# If Real Dataset Uploaded
# ---------------------------------------------

if df is not None:

    sensitive_columns = st.multiselect(
        "Select Sensitive Features",
        options=list(df.columns),
        key="real_sensitive_columns"
    )

# ---------------------------------------------
# If Synthetic Dataset Enabled
# ---------------------------------------------

elif generate_synthetic:

    st.info(
        "Synthetic dataset mode enabled."
    )

    # -----------------------------------------
    # Dynamically Extract Model Features
    # -----------------------------------------

    extracted_features = []

    if "model_features" in st.session_state:

        extracted_features = (
            st.session_state[
                "model_features"
            ]
        )

    # -----------------------------------------
    # Show Features
    # -----------------------------------------

    if len(extracted_features) > 0:

        st.write(
            "Detected Model Features:"
        )

        st.write(
            extracted_features
        )

        sensitive_columns = st.multiselect(

            "Select Sensitive Features",

            options=extracted_features,

            key="synthetic_sensitive_columns"
        )

    else:

        st.warning(
            "No model features extracted yet."
        )

# =====================================================
# Run Governance
# =====================================================

st.header("4. Run Governance")

if st.button(
    "Run Governance Analysis",
    key="run_governance_button"
):

    # -------------------------------------------------
    # Validate Model
    # -------------------------------------------------

    if "model_path" not in st.session_state:

        st.error(
            "Upload model first"
        )

    # -------------------------------------------------
    # Validate Dataset
    # -------------------------------------------------

    elif not generate_synthetic and dataset_path is None:

        st.error(
            "Upload dataset first"
        )

    else:

        # ---------------------------------------------
        # Payload
        # ---------------------------------------------

        payload = {

            "model_path":
                st.session_state[
                    "model_path"
                ],

            "dataset_path":
                dataset_path,

            "target_column":
                target_column,

            "sensitive_columns":
                sensitive_columns,

            "generate_synthetic":
                generate_synthetic,

            "synthetic_rows":
                synthetic_rows
                if generate_synthetic
                else 1000
        }

        # ---------------------------------------------
        # API Call
        # ---------------------------------------------

        with st.spinner(
            "Running governance analysis..."
        ):

            response = requests.post(
                f"{BACKEND_URL}/api/run-governance",
                json=payload
            )

        # ---------------------------------------------
        # Success
        # ---------------------------------------------

        if response.status_code == 200:

            result = response.json()

            st.success(
                "Governance analysis complete"
            )

            # =========================================
            # Governance Summary
            # =========================================

            st.subheader(
                "Governance Summary"
            )

            st.json(
                result[
                    "governance_summary"
                ]
            )

            # =========================================
            # Model Inspection
            # =========================================

            st.subheader(
                "Model Inspection"
            )

            st.json(
                result[
                    "model_inspection"
                ]
            )

            # =========================================
            # Dataset Preview
            # =========================================

            if "dataset_preview" in result:

                st.subheader(
                    "Dataset Preview"
                )

                preview_df = pd.DataFrame(
                    result[
                        "dataset_preview"
                    ]
                )

                st.dataframe(
                    preview_df
                )

            # =========================================
            # Deterministic Metrics
            # =========================================

            st.subheader(
                "Confusion Metrics"
            )

            confusion_metrics = (
                result[
                    "deterministic_metrics"
                ].get(
                    "confusion_metrics",
                    {}
                )
            )

            confusion_summary = {
                "accuracy":
                    confusion_metrics.get(
                        "accuracy"
                    ),
                "precision":
                    confusion_metrics.get(
                        "precision"
                    ),
                "recall":
                    confusion_metrics.get(
                        "recall"
                    ),
                "f1_score":
                    confusion_metrics.get(
                        "f1_score"
                    ),
                "true_positive":
                    confusion_metrics.get(
                        "true_positive"
                    ),
                "true_negative":
                    confusion_metrics.get(
                        "true_negative"
                    ),
                "false_positive":
                    confusion_metrics.get(
                        "false_positive"
                    ),
                "false_negative":
                    confusion_metrics.get(
                        "false_negative"
                    ),
                "confusion_matrix":
                    confusion_metrics.get(
                        "confusion_matrix"
                    )
            }

            st.json(
                confusion_summary
            )

            # =========================================
            # Fairness Metrics
            # =========================================

            st.subheader(
                "Fairness Metrics"
            )

            fairness_metrics = result.get(
                "fairness_metrics",
                {}
            )

            fairness_summary = {}

            for feature, metrics in (
                fairness_metrics.items()
            ):

                fairness_summary[
                    feature
                ] = {
                    "dir":
                        metrics.get(
                            "demographic_parity",
                            {}
                        ).get(
                            "disparate_impact_ratio"
                        ),
                    "dpd":
                        metrics.get(
                            "demographic_parity",
                            {}
                        ).get(
                            "demographic_parity_difference"
                        ),
                    "status":
                        metrics.get(
                            "policy_evaluation",
                            {}
                        ).get(
                            "overall_fairness_status"
                        ),
                    "reason":
                        metrics.get(
                            "fairness_unavailable_reason"
                        )
                }

            st.json(
                fairness_summary
            )

        # ---------------------------------------------
        # Failure
        # ---------------------------------------------

        else:

            st.error(
                response.text
            )
