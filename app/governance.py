from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import os
import numpy as np
import pandas as pd

from app.fairness import run_multi_fairness_analysis
from app.inspector import find_model_files, inspect_model, parse_mlmodel
from app.metrics import align_features_to_model, generate_predictions, run_all_metrics
from app.reports import generate_governance_summary
from app.synthetic_data import (
    generate_synthetic_dataset,
    generate_synthetic_dataset_from_reference,
    save_synthetic_dataset,
)
from app.utils import   load_model, sanitize_for_json, validate_sensitive_features    


router = APIRouter()


def infer_label_space(model, observed_predictions):
    model_classes = list(getattr(model, "classes_", []))

    if model_classes:
        return model_classes

    return pd.Series(observed_predictions).dropna().unique().tolist()


def find_reference_dataset(artifacts):
    candidate_paths = []

    candidate_model_paths = (
        artifacts.get("pickle_files", [])
        + artifacts.get("joblib_files", [])
        + artifacts.get("onnx_files", [])
        + artifacts.get("mlmodel_files", [])
    )

    for artifact_path in candidate_model_paths:
        artifact_root = os.path.dirname(os.path.dirname(artifact_path))
        candidate_paths.extend(
            [
                os.path.join(artifact_root, "data_transformation", "train.csv"),
                os.path.join(artifact_root, "data_transformation", "test.csv"),
                os.path.join(artifact_root, "data_ingestion", "fraud_data.csv"),
                os.path.join(artifact_root, "data", "train.csv"),
                os.path.join(artifact_root, "data", "test.csv"),
            ]
        )

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    return None


def generate_synthetic_target(model, aligned_features, task_type):
    baseline_predictions = pd.Series(model.predict(aligned_features)).reset_index(drop=True)

    if task_type == "regression":
        prediction_std = float(np.std(baseline_predictions)) if len(baseline_predictions) > 1 else 0.0
        regression_noise = np.random.normal(
            loc=0.0,
            scale=max(prediction_std * 0.05, 1e-6),
            size=len(baseline_predictions),
        )
        return baseline_predictions.astype(float) + regression_noise

    label_space = infer_label_space(model, baseline_predictions)

    if not label_space:
        return baseline_predictions

    if task_type == "binary_classification":
        primary_label = label_space[0]
        alternate_label = label_space[1] if len(label_space) > 1 else (1 if str(primary_label) == "0" else 0)
        synthetic_target = baseline_predictions.copy()

        if hasattr(model, "predict_proba"):
            try:
                probability_scores = pd.Series(model.predict_proba(aligned_features)[:, 1]).clip(0, 1)
                sampled_target = np.random.binomial(1, probability_scores.to_numpy())
                synthetic_target = pd.Series(sampled_target).map({0: primary_label, 1: alternate_label})
            except Exception:
                pass

        flip_rate = 0.20 if baseline_predictions.nunique() == 1 else 0.10
        noise_mask = np.random.rand(len(synthetic_target)) < flip_rate
        synthetic_target.loc[noise_mask] = synthetic_target.loc[noise_mask].apply(
            lambda value: alternate_label if value == primary_label else primary_label
        )

        if synthetic_target.nunique() == 1 and len(synthetic_target) > 0:
            synthetic_target.iloc[0] = alternate_label

        return synthetic_target

    if task_type == "multiclass_classification":
        synthetic_target = baseline_predictions.copy()

        if hasattr(model, "predict_proba"):
            try:
                probabilities = np.asarray(model.predict_proba(aligned_features))
                sampled_indices = [
                    np.random.choice(np.arange(probabilities.shape[1]), p=row / row.sum())
                    for row in probabilities
                ]
                mapped_labels = [label_space[index] for index in sampled_indices]
                return pd.Series(mapped_labels)
            except Exception:
                pass

        noise_mask = np.random.rand(len(synthetic_target)) < 0.10
        alternative_labels = label_space
        synthetic_target.loc[noise_mask] = synthetic_target.loc[noise_mask].apply(
            lambda value: np.random.choice([label for label in alternative_labels if label != value])
            if len(alternative_labels) > 1
            else value
        )

        return synthetic_target

    return baseline_predictions


class GovernanceRequest(BaseModel):
    model_path: str
    dataset_path: str | None = None
    target_column: str
    sensitive_columns: list[str]
    generate_synthetic: bool = False
    synthetic_rows: int = 1000


@router.post("/run-governance")
def run_governance(request: GovernanceRequest):
    try:
        artifacts = find_model_files(request.model_path)
        metadata = parse_mlmodel(artifacts["mlmodel"])
        loaded = load_model(artifacts)
        model = loaded["model"]
        inspection = inspect_model(model, metadata)
        model_task_type = inspection.get("model_type", "unknown")

        if request.generate_synthetic:
            feature_names = inspection.get("feature_names", [])

            if len(feature_names) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="Feature names could not be extracted from model",
                )

            reference_dataset_path = find_reference_dataset(artifacts)

            if reference_dataset_path:
                reference_df = pd.read_csv(reference_dataset_path)
                df = generate_synthetic_dataset_from_reference(
                    reference_df=reference_df,
                    feature_names=feature_names,
                    rows=request.synthetic_rows,
                )
            else:
                df = generate_synthetic_dataset(
                    feature_names=feature_names,
                    target_column=request.target_column,
                    rows=request.synthetic_rows,
                )

            aligned_features, _ = align_features_to_model(model, df)
            df[request.target_column] = generate_synthetic_target(
                model,
                aligned_features,
                model_task_type,
            ).values

            synthetic_dataset_path = f"generated_{request.target_column}.csv"
            save_synthetic_dataset(df, synthetic_dataset_path)
        else:
            if not request.dataset_path:
                raise HTTPException(
                    status_code=400,
                    detail="Dataset path required when synthetic generation is disabled",
                )

            if not os.path.exists(request.dataset_path):
                raise HTTPException(status_code=404, detail="Dataset not found")

            df = pd.read_csv(request.dataset_path)

        if request.target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{request.target_column}' not found")

        validate_sensitive_features(df, request.sensitive_columns)

        deterministic_metrics = run_all_metrics(
            model=model,
            df=df,
            target_column=request.target_column,
        )

        prediction_results = generate_predictions(
            model=model,
            df=df,
            target_column=request.target_column,
        )

        y_true = prediction_results["y_true"]
        y_pred = prediction_results["y_pred"]
        task_type = prediction_results["task_type"]

        fairness_metrics = run_multi_fairness_analysis(
            df=df,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_columns=request.sensitive_columns,
            task_type=task_type,
        )

        governance_summary = generate_governance_summary(
            deterministic_metrics,
            fairness_metrics,
        )

        inspection["loaded_model_format"] = loaded["model_format"]
        inspection["loaded_model_path"] = loaded["model_path"]
        inspection["evaluated_task_type"] = task_type

        response_payload = {
            "status": "success",
            "dataset_type": "synthetic" if request.generate_synthetic else "uploaded",
            "synthetic_rows": request.synthetic_rows if request.generate_synthetic else None,
            "model_metadata": metadata,
            "model_inspection": inspection,
            "deterministic_metrics": deterministic_metrics,
            "fairness_metrics": fairness_metrics,
            "dataset_preview": df.head(10).to_dict(orient="records"),
            "governance_summary": governance_summary,
        }

        return sanitize_for_json(response_payload)
    except HTTPException as http_error:
        raise http_error
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error))
