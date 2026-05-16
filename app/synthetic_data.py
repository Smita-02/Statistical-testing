import numpy as np
import pandas as pd

# =====================================================
# Detect Feature Types
# =====================================================

def detect_feature_types(
    feature_names
):

    """
    Dynamically infer feature types
    using lightweight heuristics.

    Returns:
    {
        "age": "numeric",
        "gender": "categorical"
    }
    """

    feature_types = {}

    numeric_keywords = [

        "age",
        "income",
        "salary",
        "amount",
        "score",
        "balance",
        "loan",
        "credit",
        "duration",
        "step",
        "price",
        "quantity",
        "count",
        "number",
        "rate",
        "total",
        "avg",
        "mean",
        "sum",
        "newbalance",
        "oldbalance"
    ]

    for feature in feature_names:

        feature_lower = feature.lower()

        # -------------------------------------------------
        # Numeric Detection
        # -------------------------------------------------

        if any(
            keyword in feature_lower
            for keyword in numeric_keywords
        ):

            feature_types[
                feature
            ] = "numeric"

        else:

            feature_types[
                feature
            ] = "categorical"

    return feature_types

# =====================================================
# Generate Numeric Column
# =====================================================

def generate_numeric_column(
    feature_name,
    rows
):

    """
    Generate synthetic numeric values.
    """

    feature = feature_name.lower()

    # -------------------------------------------------
    # Age-like Features
    # -------------------------------------------------

    if "age" in feature:

        return np.random.randint(
            18,
            80,
            rows
        )

    # -------------------------------------------------
    # Financial Features
    # -------------------------------------------------

    elif any(
        keyword in feature
        for keyword in [

            "income",
            "salary",
            "amount",
            "balance",
            "loan",
            "credit",
            "price"
        ]
    ):

        return np.random.uniform(
            100,
            100000,
            rows
        ).round(2)

    # -------------------------------------------------
    # Score Features
    # -------------------------------------------------

    elif any(
        keyword in feature
        for keyword in [

            "score",
            "rate"
        ]
    ):

        return np.random.uniform(
            0,
            100,
            rows
        ).round(2)

    # -------------------------------------------------
    # Generic Numeric
    # -------------------------------------------------

    else:

        return np.random.uniform(
            0,
            1000,
            rows
        ).round(2)

# =====================================================
# Generate Categorical Column
# =====================================================

def generate_categorical_column(
    feature_name,
    rows
):

    """
    Generate MODEL-SAFE categorical data.

    IMPORTANT:
    Most sklearn models require encoded
    categorical features.

    Therefore we generate encoded integers
    instead of strings like Male/Female.
    """

    # -------------------------------------------------
    # Binary Encoded Categories
    # -------------------------------------------------

    return np.random.choice(

        [0, 1],

        rows,

        p=[0.5, 0.5]
    )

# =====================================================
# Generate Synthetic Dataset
# =====================================================

def generate_synthetic_dataset(

    feature_names,

    target_column=None,

    rows=1000
):

    """
    Main synthetic dataset generator.

    Generates ONLY feature columns.
    Target column should later be
    generated using model predictions.
    """

    # -------------------------------------------------
    # Detect Feature Types
    # -------------------------------------------------

    feature_types = detect_feature_types(
        feature_names
    )

    data = {}

    # -------------------------------------------------
    # Generate Features
    # -------------------------------------------------

    for feature, feature_type in (
        feature_types.items()
    ):

        # Skip target column
        if feature == target_column:

            continue

        # ---------------------------------------------
        # Numeric Feature
        # ---------------------------------------------

        if feature_type == "numeric":

            data[feature] = (
                generate_numeric_column(
                    feature,
                    rows
                )
            )

        # ---------------------------------------------
        # Categorical Feature
        # ---------------------------------------------

        else:

            data[feature] = (
                generate_categorical_column(
                    feature,
                    rows
                )
            )

    # -------------------------------------------------
    # Create DataFrame
    # -------------------------------------------------

    df = pd.DataFrame(data)

    return df

# =====================================================
# Generate Synthetic Dataset From Reference Data
# =====================================================

def generate_synthetic_dataset_from_reference(

    reference_df,

    feature_names,

    rows=1000,

    random_state=42
):

    """
    Builds synthetic features by sampling from an
    existing transformed dataset. This produces
    model-compatible inputs more reliably than
    keyword heuristics alone.
    """

    available_features = [
        feature
        for feature in feature_names
        if feature in reference_df.columns
    ]

    if len(available_features) == 0:

        raise ValueError(
            "Reference dataset does not contain "
            "the required model feature columns"
        )

    rng = np.random.default_rng(
        random_state
    )

    sampled = (
        reference_df[
            available_features
        ]
        .sample(
            n=rows,
            replace=True,
            random_state=random_state
        )
        .reset_index(drop=True)
        .copy()
    )

    numeric_columns = sampled.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    for column in numeric_columns:

        unique_count = reference_df[
            column
        ].nunique(dropna=True)

        # Preserve one-hot and low-cardinality
        # engineered fields as-is.
        if unique_count <= 5:
            continue

        column_std = reference_df[
            column
        ].std()

        if pd.isna(column_std) or column_std == 0:
            continue

        noise = rng.normal(
            loc=0.0,
            scale=column_std * 0.03,
            size=rows
        )

        sampled[column] = (
            sampled[column].astype(float)
            + noise
        )

        if pd.api.types.is_integer_dtype(
            reference_df[column]
        ):

            sampled[column] = (
                sampled[column]
                .round()
                .astype(int)
            )

    return sampled

# =====================================================
# Save Dataset
# =====================================================

def save_synthetic_dataset(

    df,

    output_path
):

    """
    Save synthetic dataset.
    """

    df.to_csv(
        output_path,
        index=False
    )

    return output_path
