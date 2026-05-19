import numpy as np
import pandas as pd

from sklearn.base import is_classifier, is_regressor

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

from sklearn.utils.multiclass import type_of_target


# =========================================================
# SAFE FLOAT
# =========================================================

def safe_float(value):

    try:
        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return None

        return round(value, 4)

    except Exception:
        return None

def metric_explanation(metric_name, value):

    if value is None:
        return "Metric could not be computed."

    explanations = {

        "accuracy":
            (
                f"Model correctly predicted "
                f"{round(value * 100, 2)}% "
                f"of all records."
            ),

        "precision":
            (
                f"When the model predicted "
                f"positive, "
                f"{round(value * 100, 2)}% "
                f"predictions were correct."
            ),

        "recall":
            (
                f"Model identified "
                f"{round(value * 100, 2)}% "
                f"of actual positive cases."
            ),

        "f1_score":
            (
                "F1-score balances "
                "precision and recall."
            ),

        "true_positive":
            (
                f"{int(value)} positive "
                f"cases were correctly "
                f"identified."
            ),

        "true_negative":
            (
                f"{int(value)} negative "
                f"cases were correctly "
                f"identified."
            ),

        "false_positive":
            (
                f"{int(value)} negative "
                f"cases were incorrectly "
                f"predicted as positive."
            ),

        "false_negative":
            (
                f"{int(value)} positive "
                f"cases were missed "
                f"by the model."
            ),
    }

    return explanations.get(
        metric_name,
        ""
    )

# =========================================================
# CLEAN DATAFRAME
# =========================================================

def clean_feature_frame(df):

    cleaned_df = df.copy()

    for column in cleaned_df.columns:

        # BOOL -> INT
        if pd.api.types.is_bool_dtype(cleaned_df[column]):

            cleaned_df[column] = (
                cleaned_df[column]
                .astype(int)
            )

        # OBJECT -> STRING
        elif pd.api.types.is_object_dtype(cleaned_df[column]):

            cleaned_df[column] = (
                cleaned_df[column]
                .fillna("UNKNOWN")
                .astype(str)
            )

    return cleaned_df


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_expected_feature_names(model):

    if hasattr(model, "feature_names_in_"):

        return list(model.feature_names_in_)

    if hasattr(model, "named_steps"):

        for _, step in model.named_steps.items():

            if hasattr(step, "feature_names_in_"):

                return list(step.feature_names_in_)

    return []


# =========================================================
# ALIGN FEATURES TO MODEL
# =========================================================

def align_features_to_model(model, feature_df):

    cleaned_df = clean_feature_frame(
        feature_df
    )

    expected_features = (
        extract_expected_feature_names(
            model
        )
    )

    # -----------------------------------
    # NO FEATURE INFO FOUND
    # -----------------------------------

    if not expected_features:

        # FORCE NUMERIC
        for column in cleaned_df.columns:

            if (
                pd.api.types.is_object_dtype(
                    cleaned_df[column]
                )
                or str(
                    cleaned_df[column].dtype
                ) == "category"
            ):

                cleaned_df[column] = (
                    cleaned_df[column]
                    .astype("category")
                    .cat.codes
                )

        bool_columns = cleaned_df.select_dtypes(
            include=["bool"]
        ).columns

        for column in bool_columns:

            cleaned_df[column] = (
                cleaned_df[column]
                .astype(int)
            )

        cleaned_df = (
            cleaned_df
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .fillna(0)
            .astype(float)
        )

        return cleaned_df, {

            "preparation_mode":
                "raw_unvalidated",

            "expected_features":
                [],

            "input_features":
                list(cleaned_df.columns),

            "missing_features":
                [],

            "extra_features":
                []
        }

    # -----------------------------------
    # RAW FEATURES
    # -----------------------------------

    raw_columns = list(
        cleaned_df.columns
    )

    raw_missing = [

        column

        for column in expected_features

        if column not in raw_columns
    ]

    raw_extra = [

        column

        for column in raw_columns

        if column not in expected_features
    ]

    # -----------------------------------
    # DIRECT ALIGNMENT
    # -----------------------------------

    if not raw_missing:

        aligned_df = cleaned_df.reindex(
            columns=expected_features
        )

    else:

        # -----------------------------------
        # ONE HOT ENCODING
        # -----------------------------------

        categorical_columns = (
            cleaned_df.select_dtypes(
                include=[
                    "object",
                    "category"
                ]
            ).columns.tolist()
        )

        encoded_df = pd.get_dummies(

            cleaned_df,

            columns=categorical_columns,

            drop_first=False
        )

        aligned_df = encoded_df.reindex(

            columns=expected_features,

            fill_value=0
        )

    # -----------------------------------
    # FINAL TYPE CLEANING
    # -----------------------------------

    for column in aligned_df.columns:

        if (
            pd.api.types.is_object_dtype(
                aligned_df[column]
            )
            or str(
                aligned_df[column].dtype
            ) == "category"
        ):

            aligned_df[column] = (
                aligned_df[column]
                .astype("category")
                .cat.codes
            )

    bool_columns = aligned_df.select_dtypes(
        include=["bool"]
    ).columns

    for column in bool_columns:

        aligned_df[column] = (
            aligned_df[column]
            .astype(int)
        )

    aligned_df = (
        aligned_df
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
        .astype(float)
    )

    return aligned_df, {

        "preparation_mode":
            "fully_aligned",

        "expected_features":
            expected_features,

        "input_features":
            raw_columns,

        "missing_features":
            raw_missing,

        "extra_features":
            raw_extra
    }


# =========================================================
# PREPARE DATASET
# =========================================================

def prepare_dataset(model, df, target_column):

    if target_column not in df.columns:

        raise Exception(
            f"Target column '{target_column}' not found"
        )

    X_raw = df.drop(columns=[target_column])

    X, alignment_info = align_features_to_model(
        model,
        X_raw
    )

    y = df[target_column]

    return X, y, alignment_info


# =========================================================
# TASK TYPE DETECTION
# =========================================================

def detect_task_type(y_true, model=None):

    if model is not None and is_classifier(model):

        classes = list(
            getattr(model, "classes_", [])
        )

        if len(classes) == 2:
            return "binary_classification"

        if len(classes) > 2:
            return "multiclass_classification"

        return "classification"

    if model is not None and is_regressor(model):

        return "regression"

    target_kind = type_of_target(
        pd.Series(y_true).dropna()
    )

    if target_kind == "binary":
        return "binary_classification"

    if target_kind == "multiclass":
        return "multiclass_classification"

    if target_kind == "continuous":
        return "regression"

    return "unknown"


# =========================================================
# PREDICTION SCORES
# =========================================================

def extract_prediction_scores(model, X):

    if not hasattr(model, "predict_proba"):
        return None, None

    try:

        probabilities = model.predict_proba(X)

        probabilities = np.asarray(probabilities)

        class_labels = list(
            getattr(model, "classes_", [])
        )

        if probabilities.ndim == 2 and probabilities.shape[1] == 2:

            return probabilities[:, 1], class_labels

        return probabilities, class_labels

    except Exception:

        return None, None


# =========================================================
# GENERATE PREDICTIONS
# =========================================================

def generate_predictions(model, df, target_column):

    X, y_true, alignment_info = prepare_dataset(
        model,
        df,
        target_column
    )

    y_pred = model.predict(X)

    task_type = detect_task_type(
        y_true,
        model
    )

    y_score, class_labels = extract_prediction_scores(
        model,
        X
    )

    return {
        "X": X,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_prob": y_score,
        "class_labels": class_labels,
        "feature_alignment": alignment_info,
        "task_type": task_type,
    }


# =========================================================
# CONFUSION METRICS
# =========================================================

def calculate_confusion_metrics(y_true, y_pred):

    labels = np.unique(
        np.concatenate([
            np.asarray(y_true),
            np.asarray(y_pred)
        ])
    )

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels
    )

    result = {
        "accuracy": {
            "value":
                safe_float(
                 accuracy_score(
                     y_true,
                     y_pred
            )
        ),

            "explanation":
                metric_explanation(
                  "accuracy",
                   safe_float(
                      accuracy_score(
                       y_true,
                       y_pred
                )  
            )
        )
},

        "precision": {
    "value":
        safe_float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),

    "explanation":
        metric_explanation(
            "precision",
            safe_float(
                precision_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            )
        )
},

        "recall": {
    "value":
        safe_float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),

    "explanation":
        metric_explanation(
            "recall",
            safe_float(
                recall_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            )
        )
},

        "f1_score": {
    "value":
        safe_float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),

    "explanation":
        metric_explanation(
            "f1_score",
            safe_float(
                f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0
                )
            )
        )
},

        "balanced_accuracy": safe_float(
            balanced_accuracy_score(
                y_true,
                y_pred
            )
        ),

        "matthews_corrcoef": safe_float(
            matthews_corrcoef(
                y_true,
                y_pred
            )
        ),

        "labels": [
            str(label)
            for label in labels
        ],

        "confusion_matrix": cm.tolist(),
    }

    # -----------------------------------------------------
    # BINARY METRICS
    # -----------------------------------------------------

    if len(labels) == 2:

        tn, fp, fn, tp = cm.ravel()

        result["true_positive"] = {
            "value" : int(tp),
            "explanation":
                (
                    f"{int(tp)} positive cases"
                    f"were correctly identified "
                    f"by the model."
                )
        }
        result["true_negative"] = {
            "value" : int(tn),
             "explanation":
                (
                    f"{int(tn)} negative cases"
                    f" were correctly identified."
                    f" by the model."
                )
        }
        result["false_positive"] = {
            "value" : int(fp),
            "explanation":
            (
                f"{int(fp)} negative cases"
                f"were incorrectly predicted"
                f"as positive."
            )
        }
        result["false_negative"] = {
            "value": int(fn),
            "expalanation":
             (
                 f"{int(fn)} positive cases"
                 f" were missed by the model."
             )
        }
    else:

        result["true_positive"] = {
             "value": None,
             "explanation":  "True Positive is only available for binary classification."
        } 
        result["true_negative"] = {
            "value": None,
            "explanation":
        "True Negative is only available for binary classification."
        }
        result["false_positive"] = {
            "value":None,
            "explanation":
              "False Positive is only available for binary classification."
        }
        result["false_negative"] = {
            "value": None,
            "explanation": "False Negative is only available for binary classification."
        } 

    return result
# =========================================================
# AUC
# =========================================================

def calculate_auc(y_true, y_score, class_labels=None):

    if y_score is None:

        return {
            "roc_auc": None
        }

    try:

        y_score = np.asarray(y_score)

        if y_score.ndim == 1:

            auc = roc_auc_score(
                y_true,
                y_score
            )

        else:

            auc = roc_auc_score(
                y_true,
                y_score,
                multi_class="ovr",
                average="weighted",
                labels=class_labels
            )

        return {
            "roc_auc": safe_float(auc)
        }

    except Exception:

        return {
            "roc_auc": None
        }


# =========================================================
# CLASSIFICATION REPORT
# =========================================================

def generate_classification_metrics(y_true, y_pred):

    return classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )


# =========================================================
# PER CLASS ACCURACY
# =========================================================

def calculate_per_class_accuracy(y_true, y_pred):

    result = {}

    df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred
    })

    for cls in df["y_true"].unique():

        subset = df[
            df["y_true"] == cls
        ]

        result[str(cls)] = safe_float(
            (
                subset["y_true"]
                ==
                subset["y_pred"]
            ).mean()
        )

    return result


# =========================================================
# PREDICTION DISTRIBUTION
# =========================================================

def prediction_distribution(y_pred):

    unique, counts = np.unique(
        y_pred,
        return_counts=True
    )

    total = len(y_pred)

    distribution = {}

    for value, count in zip(unique, counts):

        distribution[str(value)] = {
            "count": int(count),
            "percentage": round(
                (count / total) * 100,
                2
            )
        }

    return distribution


# =========================================================
# REGRESSION METRICS
# =========================================================

def calculate_regression_metrics(y_true, y_pred):

    mse = mean_squared_error(
        y_true,
        y_pred
    )

    return {
        "mae": safe_float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),

        "mse": safe_float(mse),

        "rmse": safe_float(
            np.sqrt(mse)
        ),

        "r2_score": safe_float(
            r2_score(
                y_true,
                y_pred
            )
        ),

        "mape": safe_float(
            mean_absolute_percentage_error(
                y_true,
                y_pred
            )
        ),
    }


# =========================================================
# RUN ALL METRICS
# =========================================================

def run_all_metrics(model, df, target_column):

    prediction_results = generate_predictions(
        model,
        df,
        target_column
    )

    y_true = prediction_results["y_true"]
    y_pred = prediction_results["y_pred"]
    y_score = prediction_results["y_prob"]
    class_labels = prediction_results["class_labels"]

    task_type = prediction_results["task_type"]

    report = {
        "task_type": task_type,
        "feature_alignment": prediction_results["feature_alignment"],
    }

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    if task_type in [
        "binary_classification",
        "multiclass_classification",
        "classification"
    ]:

        report["confusion_metrics"] = calculate_confusion_metrics(
            y_true,
            y_pred
        )

        report["classification_report"] = (
            generate_classification_metrics(
                y_true,
                y_pred
            )
        )

        report["auc_metrics"] = calculate_auc(
            y_true,
            y_score,
            class_labels
        )

        report["prediction_distribution"] = (
            prediction_distribution(
                y_pred
            )
        )

        report["per_class_accuracy"] = (
            calculate_per_class_accuracy(
                y_true,
                y_pred
            )
        )

        return report

    # -----------------------------------------------------
    # REGRESSION
    # -----------------------------------------------------

    if task_type == "regression":

        report["regression_metrics"] = (
            calculate_regression_metrics(
                y_true,
                y_pred
            )
        )

        return report

    return report
