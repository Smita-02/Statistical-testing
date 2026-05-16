import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.base import is_classifier, is_regressor
from sklearn.utils.multiclass import type_of_target


def safe_float(value):
    try:
        numeric_value = float(value)
    except Exception:
        return None

    if np.isnan(numeric_value) or np.isinf(numeric_value):
        return None

    return numeric_value


def clean_feature_frame(df):
    cleaned_df = df.copy()

    for column in cleaned_df.columns:
        if pd.api.types.is_object_dtype(cleaned_df[column]) or str(cleaned_df[column].dtype) == "category":
            cleaned_df[column] = cleaned_df[column].fillna("UNKNOWN").astype(str)

    return cleaned_df


def extract_expected_feature_names(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):
        for _, step in model.named_steps.items():
            if hasattr(step, "feature_names_in_"):
                return list(step.feature_names_in_)

    return []


def align_features_to_model(model, feature_df):
    cleaned_df = clean_feature_frame(feature_df)
    expected_features = extract_expected_feature_names(model)

    if not expected_features:
        return cleaned_df, {
            "preparation_mode": "raw_unvalidated",
            "expected_features": [],
            "input_features": list(cleaned_df.columns),
            "missing_features": [],
            "extra_features": [],
        }

    raw_columns = list(cleaned_df.columns)
    raw_missing = [column for column in expected_features if column not in raw_columns]
    raw_extra = [column for column in raw_columns if column not in expected_features]

    if not raw_missing:
        return cleaned_df.reindex(columns=expected_features), {
            "preparation_mode": "raw_aligned",
            "expected_features": expected_features,
            "input_features": raw_columns,
            "missing_features": [],
            "extra_features": raw_extra,
        }

    categorical_columns = cleaned_df.select_dtypes(include=["object", "category"]).columns.tolist()
    encoded_df = pd.get_dummies(cleaned_df, columns=categorical_columns, drop_first=False)
    encoded_columns = list(encoded_df.columns)
    encoded_matches = sum(column in encoded_columns for column in expected_features)

    if encoded_matches > 0:
        encoded_missing = [column for column in expected_features if column not in encoded_columns]
        encoded_extra = [column for column in encoded_columns if column not in expected_features]
        return encoded_df.reindex(columns=expected_features, fill_value=0), {
            "preparation_mode": "one_hot_aligned",
            "expected_features": expected_features,
            "input_features": raw_columns,
            "missing_features": encoded_missing,
            "extra_features": encoded_extra,
        }

    return cleaned_df.reindex(columns=expected_features, fill_value=0), {
        "preparation_mode": "fallback_reindexed",
        "expected_features": expected_features,
        "input_features": raw_columns,
        "missing_features": raw_missing,
        "extra_features": raw_extra,
    }


def prepare_dataset(model, df, target_column):
    if target_column not in df.columns:
        raise Exception(f"Target column '{target_column}' not found")

    X_raw = df.drop(columns=[target_column])
    X, feature_alignment = align_features_to_model(model, X_raw)
    y = df[target_column]

    return X, y, feature_alignment


def detect_task_type(y_true, model=None):
    if model is not None and is_classifier(model):
        classes = list(getattr(model, "classes_", []))
        if len(classes) == 2:
            return "binary_classification"
        if len(classes) > 2:
            return "multiclass_classification"
        return "classification"

    if model is not None and is_regressor(model):
        return "regression"

    target_kind = type_of_target(pd.Series(y_true).dropna())

    if target_kind == "binary":
        return "binary_classification"

    if target_kind in {"multiclass", "multiclass-multioutput"}:
        return "multiclass_classification"

    if target_kind in {"continuous", "continuous-multioutput"}:
        return "regression"

    return "unknown"


def extract_prediction_scores(model, X):
    if not hasattr(model, "predict_proba"):
        return None, None

    try:
        probabilities = model.predict_proba(X)
    except Exception:
        return None, None

    class_labels = list(getattr(model, "classes_", [])) or None
    score_array = np.asarray(probabilities)

    if score_array.ndim == 1:
        return score_array, class_labels

    if score_array.ndim == 2 and score_array.shape[1] == 2:
        return score_array[:, 1], class_labels

    return score_array, class_labels


def generate_predictions(model, df, target_column):
    X, y_true, feature_alignment = prepare_dataset(model, df, target_column)
    y_pred = model.predict(X)
    task_type = detect_task_type(y_true, model)
    y_score, class_labels = extract_prediction_scores(model, X)

    return {
        "X": X,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_score,
        "class_labels": class_labels,
        "feature_alignment": feature_alignment,
        "task_type": task_type,
    }


def build_label_confusion_details(labels, cm):
    total = int(cm.sum())
    label_metrics = {}

    for index, label in enumerate(labels):
        true_positive = int(cm[index, index])
        false_positive = int(cm[:, index].sum() - true_positive)
        false_negative = int(cm[index, :].sum() - true_positive)
        true_negative = int(total - true_positive - false_positive - false_negative)

        label_metrics[str(label)] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "support": int(cm[index, :].sum()),
        }

    return label_metrics


def calculate_confusion_metrics(y_true, y_pred):
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    labels = np.unique(np.concatenate([y_true_array, y_pred_array]))
    cm = confusion_matrix(y_true_array, y_pred_array, labels=labels)
    per_label_metrics = build_label_confusion_details(labels, cm)

    metrics = {
        "status": "AVAILABLE",
        "accuracy": safe_float(accuracy_score(y_true_array, y_pred_array)),
        "precision": safe_float(
            precision_score(y_true_array, y_pred_array, average="weighted", zero_division=0)
        ),
        "recall": safe_float(recall_score(y_true_array, y_pred_array, average="weighted", zero_division=0)),
        "f1_score": safe_float(f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0)),
        "balanced_accuracy": safe_float(balanced_accuracy_score(y_true_array, y_pred_array)),
        "matthews_corrcoef": safe_float(matthews_corrcoef(y_true_array, y_pred_array)),
        "labels": [str(label) for label in labels],
        "confusion_matrix": cm.tolist(),
        "per_label_confusion": per_label_metrics,
    }

    if len(labels) == 2:
        negative_label, positive_label = labels[0], labels[1]
        binary_metrics = per_label_metrics[str(positive_label)]
        metrics.update(
            {
                "negative_label": str(negative_label),
                "positive_label": str(positive_label),
                "true_positive": binary_metrics["true_positive"],
                "true_negative": binary_metrics["true_negative"],
                "false_positive": binary_metrics["false_positive"],
                "false_negative": binary_metrics["false_negative"],
            }
        )

    return metrics


def build_not_applicable_confusion_metrics(task_type):
    return {
        "status": "NOT_APPLICABLE",
        "reason": f"Confusion metrics are not applicable for task type '{task_type}'",
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
        "labels": [],
        "confusion_matrix": [],
        "per_label_confusion": {},
    }


def calculate_auc(y_true, y_score, class_labels=None):
    if y_score is None:
        return {"roc_auc": None}

    y_true_array = np.asarray(y_true)
    score_array = np.asarray(y_score)

    try:
        if score_array.ndim == 1:
            auc = roc_auc_score(y_true_array, score_array)
        else:
            labels = class_labels if class_labels is not None else list(np.unique(y_true_array))
            auc = roc_auc_score(
                y_true_array,
                score_array,
                multi_class="ovr",
                average="weighted",
                labels=labels,
            )

        return {"roc_auc": safe_float(auc)}
    except Exception:
        return {"roc_auc": None}


def generate_classification_metrics(y_true, y_pred):
    return classification_report(y_true, y_pred, output_dict=True, zero_division=0)


def calculate_per_class_accuracy(y_true, y_pred):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    per_class = {}

    for cls in df["y_true"].dropna().unique():
        subset = df[df["y_true"] == cls]
        per_class[str(cls)] = safe_float((subset["y_true"] == subset["y_pred"]).mean())

    return per_class


def prediction_distribution(y_pred):
    unique, counts = np.unique(y_pred, return_counts=True)
    total = len(y_pred)
    distribution = {}

    for value, count in zip(unique, counts):
        distribution[str(value)] = {
            "count": int(count),
            "percentage": round((count / total) * 100, 2),
        }

    return distribution


def calculate_regression_metrics(y_true, y_pred):
    y_true_array = np.asarray(y_true, dtype=float)
    y_pred_array = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true_array, y_pred_array)

    return {
        "status": "AVAILABLE",
        "mae": safe_float(mean_absolute_error(y_true_array, y_pred_array)),
        "mse": safe_float(mse),
        "rmse": safe_float(np.sqrt(mse)),
        "r2_score": safe_float(r2_score(y_true_array, y_pred_array)),
        "mape": safe_float(mean_absolute_percentage_error(y_true_array, y_pred_array)),
    }


def prediction_summary(y_true, y_pred):
    return {
        "y_true_min": safe_float(np.min(y_true)),
        "y_true_max": safe_float(np.max(y_true)),
        "y_pred_min": safe_float(np.min(y_pred)),
        "y_pred_max": safe_float(np.max(y_pred)),
        "y_true_mean": safe_float(np.mean(y_true)),
        "y_pred_mean": safe_float(np.mean(y_pred)),
    }


def basic_dataset_statistics(df):
    stats = {}

    for column in df.columns:
        stats[column] = {
            "dtype": str(df[column].dtype),
            "missing_values": int(df[column].isnull().sum()),
            "unique_values": int(df[column].nunique()),
        }

        if pd.api.types.is_numeric_dtype(df[column]):
            stats[column]["mean"] = safe_float(df[column].mean())
            stats[column]["std"] = safe_float(df[column].std())
            stats[column]["min"] = safe_float(df[column].min())
            stats[column]["max"] = safe_float(df[column].max())

    return stats


def run_all_metrics(model, df, target_column):
    prediction_results = generate_predictions(model, df, target_column)
    y_true = prediction_results["y_true"]
    y_pred = prediction_results["y_pred"]
    y_score = prediction_results["y_prob"]
    class_labels = prediction_results["class_labels"]
    task_type = prediction_results["task_type"]

    report = {
        "task_type": task_type,
        "dataset_statistics": basic_dataset_statistics(df),
        "feature_alignment": prediction_results["feature_alignment"],
        "prediction_distribution": {},
        "per_class_accuracy": {},
        "classification_report": {},
        "auc_metrics": {"roc_auc": None},
        "regression_metrics": {},
        "prediction_summary": {},
    }

    if task_type in {"binary_classification", "multiclass_classification", "classification"}:
        report["confusion_metrics"] = calculate_confusion_metrics(y_true, y_pred)
        report["auc_metrics"] = calculate_auc(y_true, y_score, class_labels)
        report["classification_report"] = generate_classification_metrics(y_true, y_pred)
        report["prediction_distribution"] = prediction_distribution(y_pred)
        report["per_class_accuracy"] = calculate_per_class_accuracy(y_true, y_pred)
        return report

    if task_type == "regression":
        report["confusion_metrics"] = build_not_applicable_confusion_metrics(task_type)
        report["regression_metrics"] = calculate_regression_metrics(y_true, y_pred)
        report["prediction_summary"] = prediction_summary(y_true, y_pred)
        return report

    report["confusion_metrics"] = build_not_applicable_confusion_metrics(task_type)
    return report
