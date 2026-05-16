# app/utils.py

import os
import json
import pickle
import joblib
import logging
import traceback

import mlflow.pyfunc

# ---------------------------------------------------
# Logging Configuration
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------
# Supported Model Extensions
# ---------------------------------------------------

SUPPORTED_MODEL_EXTENSIONS = [
    ".pkl",
    ".joblib",
    ".onnx"
]

# ---------------------------------------------------
# Validate File Exists
# ---------------------------------------------------

def validate_file_exists(file_path):

    """
    Validates that a file exists.
    """

    if not os.path.exists(file_path):

        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    return True

# ---------------------------------------------------
# Load Pickle Model
# ---------------------------------------------------

def load_pickle_model(model_path):

    """
    Loads .pkl model safely.
    """

    validate_file_exists(model_path)

    logger.info(
        f"Loading pickle model: {model_path}"
    )

    try:

        model = joblib.load(model_path)

        return model

    except Exception:

        logger.warning(
            "Joblib load failed. Trying pickle..."
        )

        with open(model_path, "rb") as f:

            model = pickle.load(f)

        return model

# ---------------------------------------------------
# Load MLflow Model
# ---------------------------------------------------

def load_mlflow_model(model_dir):

    """
    Loads MLflow pyfunc model.
    """

    validate_file_exists(model_dir)

    logger.info(
        f"Loading MLflow model: {model_dir}"
    )

    model = mlflow.pyfunc.load_model(
        model_dir
    )

    return model

# ---------------------------------------------------
# Generic Model Loader
# ---------------------------------------------------

def load_model(model_path):

    """
    Automatically loads:
    - MLflow models
    - pickle models
    - joblib models
    """

    validate_file_exists(model_path)

    # ---------------------------------------------
    # MLflow Model
    # ---------------------------------------------

    if model_path.endswith("MLmodel"):

        model_dir = os.path.dirname(
            model_path
        )

        return load_mlflow_model(model_dir)

    # ---------------------------------------------
    # Pickle Models
    # ---------------------------------------------

    elif model_path.endswith(".pkl"):

        return load_pickle_model(
            model_path
        )

    # ---------------------------------------------
    # Joblib Models
    # ---------------------------------------------

    elif model_path.endswith(".joblib"):

        return joblib.load(
            model_path
        )

    raise Exception(
        f"Unsupported model format: {model_path}"
    )

# ---------------------------------------------------
# Detect Model Class Name
# ---------------------------------------------------

def get_model_class_name(model):

    """
    Returns model class name.
    """

    return model.__class__.__name__

# ---------------------------------------------------
# Extract Model Parameters
# ---------------------------------------------------

def extract_model_parameters(model):

    """
    Extracts sklearn model parameters.
    """

    try:

        if hasattr(model, "get_params"):

            return model.get_params()

    except Exception as e:

        logger.error(
            f"Failed extracting parameters: {e}"
        )

    return {}

# ---------------------------------------------------
# Detect Pipeline Components
# ---------------------------------------------------

def extract_pipeline_steps(model):

    """
    Extracts sklearn pipeline steps.
    """

    pipeline_steps = []

    if hasattr(model, "named_steps"):

        for step_name, step in (
            model.named_steps.items()
        ):

            pipeline_steps.append({

                "step_name":
                    step_name,

                "step_class":
                    step.__class__.__name__
            })

    return pipeline_steps

# ---------------------------------------------------
# Extract Model Capabilities
# ---------------------------------------------------

def extract_model_capabilities(model):

    """
    Detects supported inference methods.
    """

    capabilities = {

        "predict": False,
        "predict_proba": False,
        "decision_function": False,
        "transform": False
    }

    if hasattr(model, "predict"):
        capabilities["predict"] = True

    if hasattr(model, "predict_proba"):
        capabilities["predict_proba"] = True

    if hasattr(model, "decision_function"):
        capabilities["decision_function"] = True

    if hasattr(model, "transform"):
        capabilities["transform"] = True

    return capabilities

# ---------------------------------------------------
# Extract Feature Names
# ---------------------------------------------------

def extract_feature_names(model):

    """
    Attempts to extract feature names.
    """

    features = []

    # ---------------------------------------------
    # Standard sklearn models
    # ---------------------------------------------

    if hasattr(model, "feature_names_in_"):

        features = list(
            model.feature_names_in_
        )

    # ---------------------------------------------
    # Pipeline support
    # ---------------------------------------------

    elif hasattr(model, "named_steps"):

        for _, step in (
            model.named_steps.items()
        ):

            if hasattr(step, "feature_names_in_"):

                features = list(
                    step.feature_names_in_
                )

                break

    return features

# ---------------------------------------------------
# Model Summary
# ---------------------------------------------------

def build_model_summary(model):

    """
    Creates lightweight model summary.
    """

    summary = {

        "model_class":
            get_model_class_name(model),

        "pipeline_steps":
            extract_pipeline_steps(model),

        "capabilities":
            extract_model_capabilities(model),

        "feature_names":
            extract_feature_names(model),

        "parameters":
            extract_model_parameters(model)
    }

    return summary

# ---------------------------------------------------
# Safe JSON Serialization
# ---------------------------------------------------

def safe_json_serializer(obj):

    """
    Handles numpy/pandas serialization.
    """

    try:

        import numpy as np

        if isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, np.ndarray):
            return obj.tolist()

    except:
        pass

    return str(obj)

# ---------------------------------------------------
# Save JSON Utility
# ---------------------------------------------------

def save_json(
    data,
    output_path
):

    """
    Saves JSON safely.
    """

    with open(output_path, "w") as f:

        json.dump(
            data,
            f,
            indent=4,
            default=safe_json_serializer
        )

    logger.info(
        f"Saved JSON file: {output_path}"
    )

    return output_path

# ---------------------------------------------------
# Error Formatter
# ---------------------------------------------------

def format_exception(error):

    """
    Formats stacktrace for governance logs.
    """

    return {

        "error_type":
            type(error).__name__,

        "error_message":
            str(error),

        "traceback":
            traceback.format_exc()
    }

# ---------------------------------------------------
# Governance Logging Utility
# ---------------------------------------------------

def governance_log(message):

    """
    Standard governance logger.
    """

    logger.info(
        f"[AI GOVERNANCE] {message}"
    )

# ---------------------------------------------------
# Validate Dataset Columns
# ---------------------------------------------------

def validate_required_columns(
    df,
    required_columns
):

    """
    Ensures required columns exist.
    """

    missing_columns = []

    for col in required_columns:

        if col not in df.columns:

            missing_columns.append(col)

    if len(missing_columns) > 0:

        raise Exception(
            f"Missing columns: {missing_columns}"
        )

    return True

# ---------------------------------------------------
# Convert Object Columns
# ---------------------------------------------------

def clean_dataframe(df):

    """
    Cleans dataframe for inference.
    """

    cleaned_df = df.copy()

    for col in cleaned_df.columns:

        # Convert categorical columns
        if cleaned_df[col].dtype == "object":

            cleaned_df[col] = (
                cleaned_df[col]
                .astype(str)
                .fillna("UNKNOWN")
            )

    return cleaned_df

# ---------------------------------------------------
# Validate Sensitive Features
# ---------------------------------------------------

def validate_sensitive_features(
    df,
    sensitive_columns
):

    """
    Ensures sensitive columns exist.
    """

    validate_required_columns(
        df,
        sensitive_columns
    )

    return True

# ---------------------------------------------------
# SANITIZE JSON SERIALIZATION
# ---------------------------------------------------

import numpy as np
import pandas as pd

def sanitize_for_json(data):

    """
    Converts numpy/pandas objects
    into JSON serializable objects.
    """

    # Numpy Integer
    if isinstance(data, np.integer):

        return int(data)

    # Numpy Float
    elif isinstance(data, np.floating):

        return float(data)

    # Numpy Array
    elif isinstance(data, np.ndarray):

        return data.tolist()

    # Pandas DataFrame
    elif isinstance(data, pd.DataFrame):

        return data.to_dict(
            orient="records"
        )

    # Pandas Series
    elif isinstance(data, pd.Series):

        return data.tolist()

    # Dictionary
    elif isinstance(data, dict):

        return {
            key: sanitize_for_json(value)
            for key, value in data.items()
        }

    # List
    elif isinstance(data, list):

        return [
            sanitize_for_json(item)
            for item in data
        ]

    # Default
    return data