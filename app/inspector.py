# app/inspector.py

import json
import os

import pandas as pd
import yaml

from app.utils import (
    extract_feature_names as extract_model_feature_names
)

# ---------------------------------------------------
# FIND MODEL FILES
# ---------------------------------------------------

def find_model_files(model_dir: str):

    """
    Scans uploaded model directory
    and extracts important artifacts.
    """

    artifacts = {

        "mlmodel": None,

        "mlmodel_files": [],

        "pickle_files": [],

        "joblib_files": [],

        "onnx_files": [],

        "requirements": None,

        "conda_env": None,

        "python_env": None
    }

    # Walk all directories
    for root, dirs, files in os.walk(
        model_dir
    ):

        for file_name in files:

            full_path = os.path.join(
                root,
                file_name
            )

            # -----------------------------------------
            # MLmodel
            # -----------------------------------------

            if file_name == "MLmodel":

                artifacts[
                    "mlmodel_files"
                ].append(full_path)

                # First MLmodel
                if artifacts["mlmodel"] is None:

                    artifacts["mlmodel"] = (
                        full_path
                    )

            # -----------------------------------------
            # Pickle
            # -----------------------------------------

            elif file_name.endswith(".pkl"):

                artifacts[
                    "pickle_files"
                ].append(full_path)

            # -----------------------------------------
            # Joblib
            # -----------------------------------------

            elif file_name.endswith(".joblib"):

                artifacts[
                    "joblib_files"
                ].append(full_path)

            # -----------------------------------------
            # ONNX
            # -----------------------------------------

            elif file_name.endswith(".onnx"):

                artifacts[
                    "onnx_files"
                ].append(full_path)

            # -----------------------------------------
            # requirements.txt
            # -----------------------------------------

            elif file_name == "requirements.txt":

                artifacts[
                    "requirements"
                ] = full_path

            # -----------------------------------------
            # conda.yaml
            # -----------------------------------------

            elif file_name == "conda.yaml":

                artifacts[
                    "conda_env"
                ] = full_path

            # -----------------------------------------
            # python_env.yaml
            # -----------------------------------------

            elif file_name == "python_env.yaml":

                artifacts[
                    "python_env"
                ] = full_path

    return artifacts

# ---------------------------------------------------
# PARSE MLMODEL
# ---------------------------------------------------

def parse_mlmodel(
    mlmodel_path: str | None
):

    """
    Parses MLflow MLmodel metadata.
    """

    if not mlmodel_path:

        return {}

    if not os.path.exists(
        mlmodel_path
    ):

        raise FileNotFoundError(
            f"MLmodel file not found: "
            f"{mlmodel_path}"
        )

    with open(
        mlmodel_path,
        "r",
        encoding="utf-8"
    ) as handle:

        config = yaml.safe_load(
            handle
        ) or {}

    return {

        "artifact_path":
            config.get(
                "artifact_path"
            ),

        "flavors":
            config.get(
                "flavors",
                {}
            ),

        "run_id":
            config.get(
                "run_id"
            ),

        "utc_time_created":
            config.get(
                "utc_time_created"
            ),

        "model_uuid":
            config.get(
                "model_uuid"
            ),

        "mlflow_version":
            config.get(
                "mlflow_version"
            ),

        "signature":
            config.get(
                "signature"
            )
    }

# ---------------------------------------------------
# DETECT FRAMEWORK
# ---------------------------------------------------

def detect_framework(
    metadata: dict,
    model=None
):

    """
    Detects ML framework.
    """

    flavors = metadata.get(
        "flavors",
        {}
    ) if metadata else {}

    if "sklearn" in flavors:

        return "scikit-learn"

    if "xgboost" in flavors:

        return "xgboost"

    if "pytorch" in flavors:

        return "pytorch"

    if "tensorflow" in flavors:

        return "tensorflow"

    if "python_function" in flavors:

        return "python_function"

    model_name = (
        model.__class__.__name__.lower()
        if model is not None
        else ""
    )

    if "onnx" in model_name:

        return "onnx"

    if (
        "forest" in model_name
        or "regression" in model_name
        or "classifier" in model_name
    ):

        return "scikit-learn"

    return "unknown"

# ---------------------------------------------------
# EXTRACT FEATURE NAMES
# ---------------------------------------------------

def extract_feature_names(
    model,
    metadata=None
):

    """
    Extract feature names safely.
    """

    try:

        # sklearn models
        if hasattr(
            model,
            "feature_names_in_"
        ):

            return list(
                model.feature_names_in_
            )

        # pipeline last step
        if hasattr(
            model,
            "named_steps"
        ):

            for _, step in model.named_steps.items():

                if hasattr(
                    step,
                    "feature_names_in_"
                ):

                    return list(
                        step.feature_names_in_
                    )

    except Exception:

        pass

    # MLflow signature fallback
    try:

        signature = (
            metadata or {}
        ).get("signature")

        if isinstance(
            signature,
            dict
        ):

            inputs = signature.get(
                "inputs"
            )

            if isinstance(
                inputs,
                list
            ):

                return [

                    item.get("name")

                    for item in inputs

                    if isinstance(
                        item,
                        dict
                    )
                ]

    except Exception:

        pass

    return []

# ---------------------------------------------------
# DETECT MODEL TYPE
# ---------------------------------------------------

def detect_model_type(
    model
):

    """
    Detects:
    - binary classification
    - multiclass classification
    - regression
    """

    estimator_type = getattr(
        model,
        "_estimator_type",
        None
    )

    # -----------------------------------------
    # Classification
    # -----------------------------------------

    if estimator_type == "classifier":

        classes = list(
            getattr(
                model,
                "classes_",
                []
            )
        )

        if len(classes) == 2:

            return "binary_classification"

        if len(classes) > 2:

            return (
                "multiclass_classification"
            )

        return "classification"

    # -----------------------------------------
    # Regression
    # -----------------------------------------

    if estimator_type == "regressor":

        return "regression"

    # -----------------------------------------
    # Fallback
    # -----------------------------------------

    model_name = (
        model.__class__.__name__.lower()
    )

    if "classifier" in model_name:

        return "classification"

    if "regressor" in model_name:

        return "regression"

    if "onnx" in model_name:

        return "unknown"

    return "unknown"

# ---------------------------------------------------
# EXTRACT PREDICTION CAPABILITIES
# ---------------------------------------------------

def extract_prediction_capabilities(
    model
):

    capabilities = {

        "predict": False,

        "predict_proba": False,

        "decision_function": False
    }

    if hasattr(model, "predict"):

        capabilities["predict"] = True

    if hasattr(model, "predict_proba"):

        capabilities[
            "predict_proba"
        ] = True

    if hasattr(model, "decision_function"):

        capabilities[
            "decision_function"
        ] = True

    return capabilities

# ---------------------------------------------------
# DETECT SENSITIVE FEATURES
# ---------------------------------------------------

def detect_sensitive_features(
    feature_names
):

    """
    Detects potentially sensitive features.
    """

    sensitive_keywords = [

        "gender",
        "sex",
        "age",
        "race",
        "religion",
        "ethnicity",
        "marital",
        "disability",
        "region",
        "nationality"
    ]

    detected = []

    for feature in feature_names:

        lower_feature = feature.lower()

        for keyword in sensitive_keywords:

            if keyword in lower_feature:

                detected.append(
                    feature
                )

    return detected

# ---------------------------------------------------
# EXTRACT DATASET SCHEMA
# ---------------------------------------------------

def extract_dataset_schema(
    dataset_path
):

    """
    Extract dataset schema from CSV.
    """

    df = pd.read_csv(
        dataset_path
    )

    schema = []

    for col in df.columns:

        schema.append({

            "column_name":
                col,

            "dtype":
                str(df[col].dtype),

            "missing_values":
                int(
                    df[col]
                    .isnull()
                    .sum()
                ),

            "unique_values":
                int(
                    df[col]
                    .nunique()
                )
        })

    return schema

# ---------------------------------------------------
# FIND REFERENCE DATASET
# ---------------------------------------------------

def find_reference_dataset(
    artifacts
):

    """
    Searches extracted model folder
    for CSV datasets.

    Returns:
        dataset path OR None
    """

    possible_files = []

    # Root search folder
    search_root = None

    # Use MLmodel folder
    if artifacts.get("mlmodel"):

        search_root = os.path.dirname(
            artifacts["mlmodel"]
        )

    # Safety
    if not search_root:

        return None

    # Walk directories
    for root, dirs, files in os.walk(
        search_root
    ):

        for file in files:

            # CSV files only
            if file.endswith(".csv"):

                possible_files.append(

                    os.path.join(
                        root,
                        file
                    )
                )

    # No dataset found
    if len(possible_files) == 0:

        return None

    # Return first dataset
    return possible_files[0]

# ---------------------------------------------------
# MODEL INSPECTION
# ---------------------------------------------------

def inspect_model(model, metadata):

    """
    Safely inspect uploaded model.
    """

    try:

        features = extract_feature_names(
            model,
            metadata
        )

    except Exception:

        features = []

    try:

        framework = detect_framework(
            metadata,
            model
        )

    except Exception:

        framework = "unknown"

    try:

        model_type = detect_model_type(
            model
        )

    except Exception:

        model_type = "unknown"

    try:

        prediction_capabilities = (
            extract_prediction_capabilities(
                model
            )
        )

    except Exception:

        prediction_capabilities = {}

    try:

        sensitive_candidates = (
            detect_sensitive_features(
                features
            )
        )

    except Exception:

        sensitive_candidates = []

    return {

        "framework":
            framework,

        "model_type":
            model_type,

        "feature_names":
            features,

        "total_features":
            len(features),

        "prediction_capabilities":
            prediction_capabilities,

        "sensitive_feature_candidates":
            sensitive_candidates
    }

# ---------------------------------------------------
# SAVE INSPECTION REPORT
# ---------------------------------------------------

def save_inspection_report(
    report,
    output_path
):

    """
    Saves inspection report JSON.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            report,
            handle,
            indent=4
        )

    return output_path
