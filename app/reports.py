# app/reports.py

import datetime
import json
import os

import pandas as pd

from app.fairness import run_multi_fairness_analysis
from app.inspector import (
    find_model_files,
    inspect_model,
    parse_mlmodel
)

from app.metrics import (
    generate_predictions,
    run_all_metrics
)

from app.utils import (
    load_model
)

REPORT_DIR = "governance_reports"

os.makedirs(
    REPORT_DIR,
    exist_ok=True
)


def current_timestamp():

    return datetime.datetime.utcnow().strftime(
        "%Y-%m-%d_%H-%M-%S"
    )


def save_json_report(
    report_data,
    filename
):

    output_path = os.path.join(
        REPORT_DIR,
        filename
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as handle:

        json.dump(
            report_data,
            handle,
            indent=4,
            default=str
        )

    return output_path


def generate_governance_summary(
    deterministic_metrics,
    fairness_metrics
):

    summary = {
        "overall_status": "FAILED",
        "risk_level": "HIGH",
        "passed_checks": [],
        "failed_checks": []
    }

    passed_any_metric = False

    # ---------------------------------------------------
    # CONFUSION METRICS CHECK
    # ---------------------------------------------------

    confusion_metrics = deterministic_metrics.get(
        "confusion_metrics",
        {}
    )

    accuracy = confusion_metrics.get("accuracy")

    if accuracy is not None:

        passed_any_metric = True

        summary["passed_checks"].append(
            "Confusion metrics calculated"
        )

    else:

        summary["failed_checks"].append(
            "Confusion metrics unavailable"
        )

    # ---------------------------------------------------
    # FAIRNESS METRICS CHECK
    # ---------------------------------------------------

    for feature_name, metrics in fairness_metrics.items():

        demographic_parity = metrics.get(
            "demographic_parity",
            {}
        )

        dir_value = demographic_parity.get(
            "disparate_impact_ratio"
        )

        dpd_value = demographic_parity.get(
            "demographic_parity_difference"
        )

        # DIR
        if dir_value is not None:

            passed_any_metric = True

            summary["passed_checks"].append(
                f"{feature_name} DIR calculated"
            )

        else:

            summary["failed_checks"].append(
                f"{feature_name} DIR unavailable"
            )

        # DPD
        if dpd_value is not None:

            passed_any_metric = True

            summary["passed_checks"].append(
                f"{feature_name} DPD calculated"
            )

        else:

            summary["failed_checks"].append(
                f"{feature_name} DPD unavailable"
            )

    # ---------------------------------------------------
    # FINAL STATUS
    # ---------------------------------------------------

    if passed_any_metric:

        summary["overall_status"] = "PASSED"
        summary["risk_level"] = "LOW"

    else:

        summary["overall_status"] = "FAILED"
        summary["risk_level"] = "HIGH"

    return summary


def build_governance_report(
    model_path,
    dataset_path,
    target_column,
    sensitive_columns
):

    artifacts = find_model_files(
        model_path
    )

    metadata = parse_mlmodel(
        artifacts["mlmodel"]
    )

    loaded = load_model(
        artifacts
    )

    model = loaded["model"]

    inspection_report = inspect_model(
        model,
        metadata
    )

    if not os.path.exists(
        dataset_path
    ):

        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}"
        )

    df = pd.read_csv(
        dataset_path
    )

    deterministic_metrics = (
        run_all_metrics(

            model=model,

            df=df,

            target_column=
                target_column
        )
    )

    prediction_results = (
        generate_predictions(

            model=model,

            df=df,

            target_column=
                target_column
        )
    )

    fairness_metrics = (
        run_multi_fairness_analysis(

            df=df,

            y_true=
                prediction_results["y_true"],

            y_pred=
                prediction_results["y_pred"],

            sensitive_columns=
                sensitive_columns
        )
    )

    governance_summary = (
        generate_governance_summary(

            deterministic_metrics,

            fairness_metrics
        )
    )

    inspection_report[
        "loaded_model_format"
    ] = loaded[
        "model_format"
    ]

    inspection_report[
        "loaded_model_path"
    ] = loaded[
        "model_path"
    ]

    inspection_report[
        "evaluated_task_type"
    ] = prediction_results[
        "task_type"
    ]

    return {

        "report_metadata": {

            "generated_at":
                current_timestamp(),

            "model_path":
                model_path,

            "dataset_path":
                dataset_path
        },

        "artifacts":
            artifacts,

        "model_metadata":
            metadata,

        "model_inspection":
            inspection_report,

        "deterministic_metrics":
            deterministic_metrics,

        "fairness_metrics":
            fairness_metrics,

        "governance_summary":
            governance_summary
    }


def generate_and_save_report(
    model_path,
    dataset_path,
    target_column,
    sensitive_columns
):

    report = build_governance_report(

        model_path=model_path,

        dataset_path=dataset_path,

        target_column=target_column,

        sensitive_columns=sensitive_columns
    )

    filename = (
        f"governance_report_"
        f"{current_timestamp()}.json"
    )

    saved_path = save_json_report(
        report,
        filename
    )

    return {

        "report_path":
            saved_path,

        "report":
            report
    }


def generate_executive_summary(
    report
):

    deterministic_metrics = report.get(
        "deterministic_metrics",
        {}
    )

    task_type = deterministic_metrics.get(
        "task_type"
    )

    summary = {

        "overall_status":
            report.get(
                "governance_summary",
                {}
            ).get(
                "overall_status"
            ),

        "risk_level":
            report.get(
                "governance_summary",
                {}
            ).get(
                "risk_level"
            ),

        "framework":
            report.get(
                "model_inspection",
                {}
            ).get(
                "framework"
            ),

        "model_type":
            report.get(
                "model_inspection",
                {}
            ).get(
                "model_type"
            ),

        "total_features":
            report.get(
                "model_inspection",
                {}
            ).get(
                "total_features"
            ),

        "task_type":
            task_type,

        "fairness_checks":
            {}
    }

    # -----------------------------------
    # PRIMARY METRIC
    # -----------------------------------

    if task_type in {

        "binary_classification",
        "multiclass_classification",
        "classification"
    }:

        summary["primary_metric"] = (
            deterministic_metrics.get(
                "confusion_metrics",
                {}
            ).get(
                "accuracy"
            )
        )

        summary["primary_metric_name"] = (
            "accuracy"
        )

    elif task_type == "regression":

        summary["primary_metric"] = (
            deterministic_metrics.get(
                "regression_metrics",
                {}
            ).get(
                "r2_score"
            )
        )

        summary["primary_metric_name"] = (
            "r2_score"
        )

    # -----------------------------------
    # FAIRNESS CHECKS
    # -----------------------------------

    for feature, metrics in report.get(
        "fairness_metrics",
        {}
    ).items():

        policy = metrics.get(
            "policy_evaluation",
            {
                "overall_fairness_status":
                    "NOT_EVALUATED"
            }
        )

        demographic_parity = metrics.get(
            "demographic_parity",
            {}
        )

        summary["fairness_checks"][
            feature
        ] = {

            "status":
                policy.get(
                    "overall_fairness_status",
                    "NOT_EVALUATED"
                ),

            "disparate_impact_ratio":
                demographic_parity.get(
                    "disparate_impact_ratio"
                ),

            "demographic_parity_difference":
                demographic_parity.get(
                    "demographic_parity_difference"
                )
        }

    return summary