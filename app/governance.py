# app/governance.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

import os
import pandas as pd

from app.inspector import (
    find_model_files,
    parse_mlmodel,
    inspect_model,
    find_reference_dataset
)

from app.utils import (
    load_model,
    sanitize_for_json,
    validate_sensitive_features,
    align_features_to_model
)

from app.synthetic_data import (
    generate_synthetic_dataset,
    generate_synthetic_dataset_from_reference,
    generate_synthetic_target,
    save_synthetic_dataset
)

from app.metrics import (
    run_all_metrics,
    generate_predictions
)

from app.fairness import (
    run_multi_fairness_analysis
)

from app.reports import (
    generate_governance_summary
)

# ---------------------------------------------------
# ROUTER
# ---------------------------------------------------

router = APIRouter()

# ---------------------------------------------------
# REQUEST SCHEMA
# ---------------------------------------------------

class GovernanceRequest(BaseModel):

    # Extracted uploaded model path
    model_path: str

    # Optional dataset path
    dataset_path: str | None = None

    # Target column
    target_column: str

    # Sensitive columns
    sensitive_columns: list[str]

    # Generate synthetic dataset
    generate_synthetic: bool = False

    # Synthetic rows
    synthetic_rows: int = 1000


# ---------------------------------------------------
# RUN GOVERNANCE
# ---------------------------------------------------

@router.post("/run-governance")
def run_governance(
    request: GovernanceRequest
):

    try:

        # ---------------------------------------------------
        # FIND MODEL ARTIFACTS
        # ---------------------------------------------------

        artifacts = find_model_files(
            request.model_path
        )

        # ---------------------------------------------------
        # PARSE MLMODEL
        # ---------------------------------------------------

        metadata = parse_mlmodel(
            artifacts["mlmodel"]
        )

        # ---------------------------------------------------
        # LOAD MODEL
        # ---------------------------------------------------

        loaded = load_model(
            artifacts
        )

        model = loaded["model"]

        # ---------------------------------------------------
        # INSPECT MODEL
        # ---------------------------------------------------

        inspection = inspect_model(
            model,
            metadata
        )

        model_task_type = inspection.get(
            "model_type",
            "unknown"
        )

        # ---------------------------------------------------
        # DATASET HANDLING
        # ---------------------------------------------------

        if request.generate_synthetic:

            feature_names = inspection.get(
                "feature_names",
                []
            )

            if len(feature_names) == 0:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Feature names could not "
                        "be extracted from model"
                    )
                )

            # ---------------------------------------------
            # TRY FINDING REFERENCE DATASET
            # ---------------------------------------------

            reference_dataset_path = (
                find_reference_dataset(
                    artifacts
                )
            )

            # ---------------------------------------------
            # GENERATE FROM REFERENCE
            # ---------------------------------------------

            if reference_dataset_path:

                reference_df = pd.read_csv(
                    reference_dataset_path
                )

                df = (
                    generate_synthetic_dataset_from_reference(

                        reference_df=
                            reference_df,

                        feature_names=
                            feature_names,

                        rows=
                            request.synthetic_rows
                    )
                )

            # ---------------------------------------------
            # GENERATE BASIC SYNTHETIC
            # ---------------------------------------------

            else:

                df = generate_synthetic_dataset(

                    feature_names=
                        feature_names,

                    target_column=
                        request.target_column,

                    rows=
                        request.synthetic_rows
                )

            # ---------------------------------------------
            # ALIGN FEATURES
            # ---------------------------------------------

            aligned_features, aligned_df = (
                align_features_to_model(
                    model,
                    df
                )
            )

            df[request.target_column] = generate_synthetic_target(
            model=model,
            X=aligned_features,
            task_type=model_task_type
            )
            # ---------------------------------------------
            # SAVE GENERATED DATASET
            # ---------------------------------------------

            synthetic_dataset_path = (

                f"generated_"
                f"{request.target_column}.csv"
            )

            save_synthetic_dataset(
                df,
                synthetic_dataset_path
            )

        # ---------------------------------------------------
        # USE UPLOADED DATASET
        # ---------------------------------------------------

        else:

            if not request.dataset_path:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Dataset path required "
                        "when synthetic generation "
                        "is disabled"
                    )
                )

            if not os.path.exists(
                request.dataset_path
            ):

                raise HTTPException(
                    status_code=404,
                    detail="Dataset not found"
                )

            df = pd.read_csv(
                request.dataset_path
            )

        # ---------------------------------------------------
        # VALIDATE TARGET
        # ---------------------------------------------------

        if request.target_column not in df.columns:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Target column "
                    f"'{request.target_column}' "
                    f"not found"
                )
            )

        # ---------------------------------------------------
        # VALIDATE SENSITIVE FEATURES
        # ---------------------------------------------------

        validate_sensitive_features(
            df,
            request.sensitive_columns
        )

        # ---------------------------------------------------
        # DETERMINE METRICS
        # ---------------------------------------------------

        deterministic_metrics = (
            run_all_metrics(

                model=model,

                df=df,

                target_column=
                    request.target_column
            )
        )

        # ---------------------------------------------------
        # GENERATE PREDICTIONS
        # ---------------------------------------------------

        prediction_results = (
            generate_predictions(

                model=model,

                df=df,

                target_column=
                    request.target_column
            )
        )

        y_true = prediction_results[
            "y_true"
        ]

        y_pred = prediction_results[
            "y_pred"
        ]

        task_type = prediction_results[
            "task_type"
        ]

        # ---------------------------------------------------
        # FAIRNESS ANALYSIS
        # ---------------------------------------------------

        fairness_metrics = (
            run_multi_fairness_analysis(

                df=df,

                y_true=y_true,

                y_pred=y_pred,

                sensitive_columns=
                    request.sensitive_columns
            )
        )

        # ---------------------------------------------------
        # GOVERNANCE SUMMARY
        # ---------------------------------------------------

        governance_summary = (
            generate_governance_summary(

                deterministic_metrics,

                fairness_metrics
            )
        )

        # ---------------------------------------------------
        # EXTRA INSPECTION INFO
        # ---------------------------------------------------

        inspection[
            "loaded_model_format"
        ] = loaded[
            "model_format"
        ]

        inspection[
            "loaded_model_path"
        ] = loaded[
            "model_path"
        ]

        inspection[
            "evaluated_task_type"
        ] = task_type

        # ---------------------------------------------------
        # RESPONSE
        # ---------------------------------------------------

        response_payload = {

            "status":
                "success",

            "dataset_type":

                "synthetic"

                if request.generate_synthetic

                else "uploaded",

            "synthetic_rows":

                request.synthetic_rows

                if request.generate_synthetic

                else None,

            "model_metadata":
                metadata,

            "model_inspection":
                inspection,

            "deterministic_metrics":
                deterministic_metrics,

            "fairness_metrics":
                fairness_metrics,

            "dataset_preview":
                df.head(10).to_dict(
                    orient="records"
                ),

            "governance_summary":
                governance_summary
        }

        return sanitize_for_json(
            response_payload
        )

    # ---------------------------------------------------
    # HTTP ERRORS
    # ---------------------------------------------------

    except HTTPException as http_error:

        raise http_error

    # ---------------------------------------------------
    # GENERAL ERRORS
    # ---------------------------------------------------

    except Exception as error:

        raise HTTPException(

            status_code=500,

            detail=str(error)
        )