import numpy as np
import pandas as pd

from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    selection_rate,
)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def safe_metric_value(value):
    """
    Converts fairness outputs to JSON-safe floats.
    """

    numeric_value = float(value)

    if np.isnan(numeric_value) or np.isinf(numeric_value):
        return None

    return numeric_value


def validate_sensitive_feature(df, sensitive_column):
    """
    Ensures the requested sensitive column exists.
    """

    if sensitive_column not in df.columns:
        raise Exception(f"Sensitive column '{sensitive_column}' not found")

    return True


def prepare_sensitive_feature(df, sensitive_column):
    """
    Normalizes the sensitive feature for grouping.
    """

    validate_sensitive_feature(df, sensitive_column)

    sensitive_series = df[
        sensitive_column
    ]

    if pd.api.types.is_numeric_dtype(
        sensitive_series
    ) and sensitive_series.nunique(
        dropna=True
    ) > 10:

        try:

            bucketed = pd.qcut(
                sensitive_series,
                q=4,
                duplicates="drop"
            )

            return bucketed.astype(str).fillna(
                "UNKNOWN"
            )

        except Exception:
            pass

    return sensitive_series.fillna(
        "UNKNOWN"
    ).astype(str)


def prepare_fairness_labels(y_true, y_pred):
    """
    Fairlearn parity metrics assume binary labels.
    This maps the observed classes dynamically
    instead of assuming fixed hardcoded labels.
    """

    y_true_series = pd.Series(y_true).reset_index(drop=True)
    y_pred_series = pd.Series(y_pred).reset_index(drop=True)
    observed_labels = pd.Index(
        pd.concat([y_true_series, y_pred_series]).dropna().unique()
    ).tolist()

    if len(observed_labels) != 2:
        raise ValueError(
            "Fairness metrics currently require binary classification outputs. "
            f"Observed labels: {observed_labels}"
        )

    label_mapping = {
        label: index for index, label in enumerate(observed_labels)
    }

    return (
        y_true_series.map(label_mapping).astype(int),
        y_pred_series.map(label_mapping).astype(int),
        {str(label): mapped for label, mapped in label_mapping.items()},
    )


def build_unavailable_fairness_report(df, y_true, y_pred, sensitive_column, reason):
    """
    Returns a structured fairness response when
    binary fairness metrics cannot be computed.
    """

    sensitive_features = prepare_sensitive_feature(df, sensitive_column)
    observed_labels = (
        pd.Index(
            pd.concat(
                [
                    pd.Series(y_true).reset_index(drop=True),
                    pd.Series(y_pred).reset_index(drop=True),
                ]
            ).dropna().unique()
        ).tolist()
    )

    return {
        "sensitive_feature": sensitive_column,
        "distribution": sensitive_feature_distribution(sensitive_features),
        "label_mapping": {},
        "fairness_status": "UNAVAILABLE",
        "fairness_unavailable_reason": reason,
        "observed_labels": [str(label) for label in observed_labels],
        "demographic_parity": {
            "demographic_parity_difference": None,
            "disparate_impact_ratio": None,
        },
        "equalized_odds": {
            "equalized_odds_difference": None,
            "equalized_odds_ratio": None,
        },
        "selection_rates": {},
        "group_metrics": {},
        "policy_evaluation": {
            "overall_fairness_status": "NOT_EVALUATED",
            "violations": [reason],
        },
    }


def calculate_demographic_parity(y_true, y_pred, sensitive_features):
    """
    Computes demographic parity metrics.
    """

    dp_difference = demographic_parity_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    dp_ratio = demographic_parity_ratio(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    return {
        "demographic_parity_difference": safe_metric_value(dp_difference),
        "disparate_impact_ratio": safe_metric_value(dp_ratio),
    }


def calculate_equalized_odds(y_true, y_pred, sensitive_features):
    """
    Computes equalized odds metrics.
    """

    eo_difference = equalized_odds_difference(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )
    eo_ratio = equalized_odds_ratio(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    return {
        "equalized_odds_difference": safe_metric_value(eo_difference),
        "equalized_odds_ratio": safe_metric_value(eo_ratio),
    }


def calculate_selection_rates(y_true, y_pred, sensitive_features):
    """
    Calculates positive prediction rates by group.
    """

    metric_frame = MetricFrame(
        metrics=selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    return {
        str(group): (
            round(safe_metric_value(value), 4)
            if safe_metric_value(value) is not None
            else None
        )
        for group, value in metric_frame.by_group.items()
    }


def calculate_group_metrics(y_true, y_pred, sensitive_features):
    """
    Computes classification metrics separately by group.
    """

    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
            "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
            "f1_score": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
    )

    results = {}

    for metric_name, metric_values in metric_frame.by_group.items():
        results[metric_name] = {
            str(group): (
                round(safe_metric_value(value), 4)
                if safe_metric_value(value) is not None
                else None
            )
            for group, value in metric_values.items()
        }

    return results


def evaluate_fairness_thresholds(fairness_metrics):
    """
    Applies governance policy thresholds.
    """

    status = {"overall_fairness_status": "PASSED", "violations": []}

    if fairness_metrics["disparate_impact_ratio"] < 0.80:
        status["overall_fairness_status"] = "FAILED"
        status["violations"].append("Disparate Impact Ratio below 0.80")

    if abs(fairness_metrics["demographic_parity_difference"]) > 0.10:
        status["overall_fairness_status"] = "FAILED"
        status["violations"].append("Demographic Parity Difference exceeds 0.10")

    return status


def sensitive_feature_distribution(sensitive_features):
    """
    Computes the group distribution.
    """

    counts = sensitive_features.value_counts(dropna=False)
    total = counts.sum()

    distribution = {}

    for group, count in counts.items():
        distribution[str(group)] = {
            "count": int(count),
            "percentage": round((count / total) * 100, 2),
        }

    return distribution


def run_fairness_analysis(df, y_true, y_pred, sensitive_column):
    """
    Executes a full fairness analysis for one
    sensitive feature.
    """

    sensitive_features = prepare_sensitive_feature(df, sensitive_column)

    try:
        encoded_y_true, encoded_y_pred, label_mapping = prepare_fairness_labels(y_true, y_pred)
    except ValueError as error:
        return build_unavailable_fairness_report(
            df=df,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_column=sensitive_column,
            reason=str(error),
        )

    parity_metrics = calculate_demographic_parity(
        encoded_y_true,
        encoded_y_pred,
        sensitive_features,
    )
    equalized_metrics = calculate_equalized_odds(
        encoded_y_true,
        encoded_y_pred,
        sensitive_features,
    )
    selection_rates = calculate_selection_rates(
        encoded_y_true,
        encoded_y_pred,
        sensitive_features,
    )
    group_metrics = calculate_group_metrics(
        encoded_y_true,
        encoded_y_pred,
        sensitive_features,
    )
    distribution = sensitive_feature_distribution(sensitive_features)
    policy_evaluation = evaluate_fairness_thresholds(parity_metrics)

    return {
        "sensitive_feature": sensitive_column,
        "distribution": distribution,
        "label_mapping": label_mapping,
        "demographic_parity": parity_metrics,
        "equalized_odds": equalized_metrics,
        "selection_rates": selection_rates,
        "group_metrics": group_metrics,
        "policy_evaluation": policy_evaluation,
    }


def run_multi_fairness_analysis(df, y_true, y_pred, sensitive_columns):
    """
    Runs fairness analysis for multiple sensitive features.
    """

    results = {}

    for sensitive_column in sensitive_columns:
        results[sensitive_column] = run_fairness_analysis(
            df=df,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_column=sensitive_column,
        )

    return results
