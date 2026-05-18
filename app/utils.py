# app/utils.py

import json
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import mlflow.sklearn

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

def load_model(artifacts: dict):

    """
    Loads ML model from extracted artifacts.

    Supported:
    - MLflow
    - Pickle
    - Joblib
    """

    try:

        # ---------------------------------------------
        # MLflow Model
        # ---------------------------------------------

        if artifacts.get("mlmodel"):

            mlmodel_path = os.path.dirname(
                artifacts["mlmodel"]
            )

            model = mlflow.sklearn.load_model(
                mlmodel_path
            )

            return {

                "model": model,

                "model_format": "mlflow",

                "model_path": mlmodel_path
            }

        # ---------------------------------------------
        # Pickle Model
        # ---------------------------------------------

        elif len(
            artifacts.get(
                "pickle_files",
                []
            )
        ) > 0:

            model_path = artifacts[
                "pickle_files"
            ][0]

            with open(
                model_path,
                "rb"
            ) as file:

                model = pickle.load(file)

            return {

                "model": model,

                "model_format": "pickle",

                "model_path": model_path
            }

        # ---------------------------------------------
        # Joblib Model
        # ---------------------------------------------

        elif len(
            artifacts.get(
                "joblib_files",
                []
            )
        ) > 0:

            model_path = artifacts[
                "joblib_files"
            ][0]

            model = joblib.load(
                model_path
            )

            return {

                "model": model,

                "model_format": "joblib",

                "model_path": model_path
            }

        # ---------------------------------------------
        # Unsupported
        # ---------------------------------------------

        else:

            raise Exception(
                "No supported model file found"
            )

    except Exception as error:

        raise Exception(
            f"Model loading failed: {str(error)}"
        )

# ---------------------------------------------------
# VALIDATE SENSITIVE FEATURES
# ---------------------------------------------------

def validate_sensitive_features(
    df,
    sensitive_columns
):

    """
    Ensures selected sensitive columns
    exist in dataset.
    """

    missing_columns = []

    for column in sensitive_columns:

        if column not in df.columns:

            missing_columns.append(column)

    if len(missing_columns) > 0:

        raise Exception(
            f"Missing columns: {missing_columns}"
        )

# ---------------------------------------------------
# EXTRACT FEATURE NAMES
# ---------------------------------------------------

def extract_feature_names(model):

    """
    Extracts feature names from model.
    """

    # ---------------------------------------------
    # sklearn >= 1.0
    # ---------------------------------------------

    if hasattr(model, "feature_names_in_"):

        return list(
            model.feature_names_in_
        )

    # ---------------------------------------------
    # Pipeline support
    # ---------------------------------------------

    if hasattr(model, "named_steps"):

        for _, step in model.named_steps.items():

            if hasattr(step, "feature_names_in_"):

                return list(
                    step.feature_names_in_
                )

    # ---------------------------------------------
    # Booster / XGBoost support
    # ---------------------------------------------

    if hasattr(model, "get_booster"):

        try:

            booster = model.get_booster()

            return booster.feature_names

        except:

            pass

    return []

# ---------------------------------------------------
# ALIGN FEATURES TO MODEL
# ---------------------------------------------------

def align_features_to_model(
    model,
    df
):

    """
    Aligns dataframe columns to model features.
    """

    model_features = extract_feature_names(
        model
    )

    if len(model_features) == 0:

        return df.copy(), []

    aligned_df = df.copy()

    # ---------------------------------------------
    # Add Missing Features
    # ---------------------------------------------

    for feature in model_features:

        if feature not in aligned_df.columns:

            aligned_df[feature] = 0

    # ---------------------------------------------
    # Remove Extra Features
    # ---------------------------------------------

    aligned_df = aligned_df[
        model_features
    ]

    return aligned_df, model_features

# ---------------------------------------------------
# SANITIZE FOR JSON
# ---------------------------------------------------

def sanitize_for_json(data):

    """
    Converts numpy/pandas objects
    into JSON serializable types.
    """

    # ---------------------------------------------
    # Dictionary
    # ---------------------------------------------

    if isinstance(data, dict):

        return {

            str(key): sanitize_for_json(value)

            for key, value in data.items()
        }

    # ---------------------------------------------
    # List / Tuple
    # ---------------------------------------------

    elif isinstance(data, (list, tuple)):

        return [

            sanitize_for_json(item)

            for item in data
        ]

    # ---------------------------------------------
    # NumPy Integer
    # ---------------------------------------------

    elif isinstance(data, np.integer):

        return int(data)

    # ---------------------------------------------
    # NumPy Float
    # ---------------------------------------------

    elif isinstance(data, np.floating):

        return float(data)

    # ---------------------------------------------
    # NumPy Array
    # ---------------------------------------------

    elif isinstance(data, np.ndarray):

        return data.tolist()

    # ---------------------------------------------
    # Pandas DataFrame
    # ---------------------------------------------

    elif isinstance(data, pd.DataFrame):

        return data.to_dict(
            orient="records"
        )

    # ---------------------------------------------
    # Pandas Series
    # ---------------------------------------------

    elif isinstance(data, pd.Series):

        return data.tolist()

    # ---------------------------------------------
    # NaN
    # ---------------------------------------------

    elif pd.isna(data):

        return None

    return data

# ---------------------------------------------------
# SAVE JSON REPORT
# ---------------------------------------------------

def save_json_report(
    report,
    output_path
):

    """
    Saves governance report.
    """

    sanitized_report = sanitize_for_json(
        report
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            sanitized_report,
            file,
            indent=4
        )

    return output_path