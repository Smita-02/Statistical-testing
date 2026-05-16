import datetime
import json
import os

import pandas as pd

from app.fairness import run_multi_fairness_analysis
from app.inspector import find_model_files, inspect_model, parse_mlmodel
from app.metrics import generate_predictions, run_all_metrics
from app.utils import load_supported_model


REPORT_DIR = "governance_reports"
os.makedirs(REPORT_DIR, exist_ok=True)


def current_timestamp():
    return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def save_json_report(report_data, filename):
    output_path = os.path.join(REPORT_DIR, filename)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=4, default=str)

    return output_path


def generate_governance_summary(deterministic_metrics, fairness_metrics):
    task_type = deterministic_metrics.get("task_type", "unknown")
    summary = {
        "overall_status": "PASSED",
        "risk_level": "LOW",
        "violations": [],
        "fairness_status": "PASSED",
        "task_type": task_type,
    }

    if task_type in {"binary_classification", "multiclass_classification", "classification"}:
        confusion_metrics = deterministic_metrics["confusion_metrics"]
        accuracy = confusion_metrics.get("accuracy")

        if accuracy is not None and accuracy < 0.70:
            summary["overall_status"] = "FAILED"
            summary["risk_level"] = "HIGH"
            summary["violations"].append("Model accuracy below threshold")

        true_positive = confusion_metrics.get("true_positive")
        false_negative = confusion_metrics.get("false_negative")

        if (
            true_positive is not None
            and false_negative is not None
            and true_positive == 0
            and false_negative > 0
        ):
            summary["overall_status"] = "FAILED"
            summary["risk_level"] = "HIGH"
            summary["violations"].append("Model failed to identify any positive cases")

    elif task_type == "regression":
        regression_metrics = deterministic_metrics.get("regression_metrics", {})
        r2_score = regression_metrics.get("r2_score")

        if r2_score is not None and r2_score < 0.50:
            summary["overall_status"] = "FAILED"
            summary["risk_level"] = "HIGH"
            summary["violations"].append("Regression R2 score below threshold")

    fairness_failures = 0
    fairness_evaluated = 0

    for feature, report in fairness_metrics.items():
        fairness_status = report["policy_evaluation"]["overall_fairness_status"]

        if fairness_status in {"PASSED", "FAILED"}:
            fairness_evaluated += 1

        if fairness_status == "FAILED":
            fairness_failures += 1
            summary["violations"].append(f"Fairness violation in {feature}")

    if fairness_failures > 0:
        summary["fairness_status"] = "FAILED"
    elif fairness_evaluated == 0:
        summary["fairness_status"] = "NOT_EVALUATED"

    if fairness_failures >= 2:
        summary["risk_level"] = "HIGH"
        summary["overall_status"] = "FAILED"
    elif fairness_failures == 1 and summary["risk_level"] == "LOW":
        summary["risk_level"] = "MEDIUM"

    if summary["fairness_status"] == "PASSED":
        summary["overall_status"] = "PASSED"

    return summary


def build_governance_report(model_path, dataset_path, target_column, sensitive_columns):
    artifacts = find_model_files(model_path)
    metadata = parse_mlmodel(artifacts["mlmodel"])
    loaded = load_supported_model(artifacts)
    model = loaded["model"]
    inspection_report = inspect_model(model, metadata)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    df = pd.read_csv(dataset_path)
    deterministic_metrics = run_all_metrics(model=model, df=df, target_column=target_column)
    prediction_results = generate_predictions(model=model, df=df, target_column=target_column)
    fairness_metrics = run_multi_fairness_analysis(
        df=df,
        y_true=prediction_results["y_true"],
        y_pred=prediction_results["y_pred"],
        sensitive_columns=sensitive_columns,
        task_type=prediction_results["task_type"],
    )
    governance_summary = generate_governance_summary(deterministic_metrics, fairness_metrics)

    inspection_report["loaded_model_format"] = loaded["model_format"]
    inspection_report["loaded_model_path"] = loaded["model_path"]
    inspection_report["evaluated_task_type"] = prediction_results["task_type"]

    return {
        "report_metadata": {
            "generated_at": current_timestamp(),
            "model_path": model_path,
            "dataset_path": dataset_path,
        },
        "artifacts": artifacts,
        "model_metadata": metadata,
        "model_inspection": inspection_report,
        "deterministic_metrics": deterministic_metrics,
        "fairness_metrics": fairness_metrics,
        "governance_summary": governance_summary,
    }


def generate_and_save_report(model_path, dataset_path, target_column, sensitive_columns):
    report = build_governance_report(
        model_path=model_path,
        dataset_path=dataset_path,
        target_column=target_column,
        sensitive_columns=sensitive_columns,
    )

    filename = f"governance_report_{current_timestamp()}.json"
    saved_path = save_json_report(report, filename)

    return {"report_path": saved_path, "report": report}


def generate_executive_summary(report):
    deterministic_metrics = report["deterministic_metrics"]
    task_type = deterministic_metrics.get("task_type")

    summary = {
        "overall_status": report["governance_summary"]["overall_status"],
        "risk_level": report["governance_summary"]["risk_level"],
        "framework": report["model_inspection"]["framework"],
        "model_type": report["model_inspection"]["model_type"],
        "total_features": report["model_inspection"]["total_features"],
        "task_type": task_type,
        "fairness_checks": {},
    }

    if task_type in {"binary_classification", "multiclass_classification", "classification"}:
        summary["primary_metric"] = deterministic_metrics["confusion_metrics"].get("accuracy")
        summary["primary_metric_name"] = "accuracy"
    elif task_type == "regression":
        summary["primary_metric"] = deterministic_metrics["regression_metrics"].get("r2_score")
        summary["primary_metric_name"] = "r2_score"

    for feature, metrics in report["fairness_metrics"].items():
        summary["fairness_checks"][feature] = {
            "status": metrics["policy_evaluation"]["overall_fairness_status"],
            "disparate_impact_ratio": metrics["demographic_parity"]["disparate_impact_ratio"],
        }

    return summary
