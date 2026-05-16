import json
import os

import pandas as pd
import yaml

from app.utils import extract_feature_names as extract_model_feature_names


def find_model_files(model_dir: str):
    artifacts = {
        "mlmodel": None,
        "mlmodel_files": [],
        "pickle_files": [],
        "joblib_files": [],
        "onnx_files": [],
        "requirements": None,
        "conda_env": None,
        "python_env": None,
    }

    for root, _, files in os.walk(model_dir):
        for file_name in files:
            full_path = os.path.join(root, file_name)

            if file_name == "MLmodel":
                artifacts["mlmodel_files"].append(full_path)
                if artifacts["mlmodel"] is None:
                    artifacts["mlmodel"] = full_path
            elif file_name.endswith(".pkl"):
                artifacts["pickle_files"].append(full_path)
            elif file_name.endswith(".joblib"):
                artifacts["joblib_files"].append(full_path)
            elif file_name.endswith(".onnx"):
                artifacts["onnx_files"].append(full_path)
            elif file_name == "requirements.txt":
                artifacts["requirements"] = full_path
            elif file_name == "conda.yaml":
                artifacts["conda_env"] = full_path
            elif file_name == "python_env.yaml":
                artifacts["python_env"] = full_path

    return artifacts


def parse_mlmodel(mlmodel_path: str | None):
    if not mlmodel_path:
        return {}

    if not os.path.exists(mlmodel_path):
        raise FileNotFoundError(f"MLmodel file not found: {mlmodel_path}")

    with open(mlmodel_path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    return {
        "artifact_path": config.get("artifact_path"),
        "flavors": config.get("flavors", {}),
        "run_id": config.get("run_id"),
        "utc_time_created": config.get("utc_time_created"),
        "model_uuid": config.get("model_uuid"),
        "mlflow_version": config.get("mlflow_version"),
        "signature": config.get("signature"),
    }


def detect_framework(metadata: dict, model=None):
    flavors = metadata.get("flavors", {}) if metadata else {}

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

    model_name = model.__class__.__name__.lower() if model is not None else ""

    if "onnx" in model_name:
        return "onnx"
    if "forest" in model_name or "regression" in model_name or "classifier" in model_name:
        return "scikit-learn"

    return "unknown"


def extract_feature_names(model, metadata=None):
    feature_names = extract_model_feature_names(model)

    if feature_names:
        return feature_names

    signature = (metadata or {}).get("signature")

    if isinstance(signature, dict):
        inputs = signature.get("inputs")
        if isinstance(inputs, list):
            return [item.get("name") for item in inputs if isinstance(item, dict) and item.get("name")]

        if isinstance(inputs, str):
            try:
                parsed_inputs = json.loads(inputs)
                if isinstance(parsed_inputs, list):
                    return [item.get("name") for item in parsed_inputs if isinstance(item, dict) and item.get("name")]
            except Exception:
                return []

    return []


def detect_model_type(model):
    estimator_type = getattr(model, "_estimator_type", None)

    if estimator_type == "classifier":
        classes = list(getattr(model, "classes_", []))
        if len(classes) == 2:
            return "binary_classification"
        if len(classes) > 2:
            return "multiclass_classification"
        return "classification"

    if estimator_type == "regressor":
        return "regression"

    model_name = model.__class__.__name__.lower()

    if "classifier" in model_name:
        return "classification"
    if "regressor" in model_name:
        return "regression"
    if "onnx" in model_name:
        return "unknown"

    return "unknown"


def extract_prediction_capabilities(model):
    capabilities = {
        "predict": False,
        "predict_proba": False,
        "decision_function": False,
    }

    if hasattr(model, "predict"):
        capabilities["predict"] = True
    if hasattr(model, "predict_proba"):
        capabilities["predict_proba"] = True
    if hasattr(model, "decision_function"):
        capabilities["decision_function"] = True

    return capabilities


def detect_sensitive_features(feature_names):
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
        "nationality",
    ]

    detected = []

    for feature in feature_names:
        lower_feature = feature.lower()
        for keyword in sensitive_keywords:
            if keyword in lower_feature:
                detected.append(feature)

    return detected


def extract_dataset_schema(dataset_path):
    df = pd.read_csv(dataset_path)
    schema = []

    for col in df.columns:
        schema.append(
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "missing_values": int(df[col].isnull().sum()),
                "unique_values": int(df[col].nunique()),
            }
        )

    return schema


def inspect_model(model, metadata):
    features = extract_feature_names(model, metadata)

    return {
        "framework": detect_framework(metadata, model),
        "model_type": detect_model_type(model),
        "feature_names": features,
        "total_features": len(features),
        "prediction_capabilities": extract_prediction_capabilities(model),
        "sensitive_feature_candidates": detect_sensitive_features(features),
    }


def save_inspection_report(report, output_path):
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=4)

    return output_path
