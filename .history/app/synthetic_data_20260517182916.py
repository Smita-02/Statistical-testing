import random
import numpy as np
import pandas as pd

# from faker import Faker

# fake = Faker()


def generate_numeric_column(rows, min_value=0, max_value=1000):
    return np.random.randint(min_value, max_value, rows)


def generate_float_column(rows, min_value=0, max_value=1000):
    return np.random.uniform(min_value, max_value, rows)


def generate_binary_column(rows):
    return np.random.randint(0, 2, rows)


def generate_categorical_column(rows, categories=None):
    if categories is None:
        categories = ["A", "B", "C"]

    return np.random.choice(categories, rows)


def generate_text_column(rows):
    return [fake.city() for _ in range(rows)]


def detect_column_type(column_name):
    column = column_name.lower()

    if any(keyword in column for keyword in [
        "age",
        "amt",
        "amount",
        "income",
        "salary",
        "score",
        "balance",
        "population",
        "city_pop",
        "duration",
        "count",
        "number"
    ]):
        return "numeric"

    if any(keyword in column for keyword in [
        "gender",
        "sex",
        "flag",
        "is_",
        "_flag"
    ]):
        return "binary"

    if any(keyword in column for keyword in [
        "category",
        "type",
        "class",
        "segment",
        "group"
    ]):
        return "categorical"

    return "text"


def generate_synthetic_dataset(
    feature_names,
    target_column,
    rows=1000
):
    data = {}

    for feature in feature_names:

        feature_type = detect_column_type(feature)

        # -----------------------------------
        # NUMERIC
        # -----------------------------------

        if feature_type == "numeric":

            data[feature] = generate_numeric_column(
                rows=rows,
                min_value=1,
                max_value=10000
            )

        # -----------------------------------
        # BINARY
        # -----------------------------------

        elif feature_type == "binary":

            data[feature] = generate_binary_column(
                rows=rows
            )

        # -----------------------------------
        # CATEGORICAL
        # -----------------------------------

        elif feature_type == "categorical":

            categories = ["A", "B", "C"]

            data[feature] = generate_categorical_column(
                rows=rows,
                categories=categories
            )

        # -----------------------------------
        # TEXT
        # -----------------------------------

        else:

            data[feature] = generate_text_column(
                rows=rows
            )

    synthetic_df = pd.DataFrame(data)

    # -----------------------------------
    # CONVERT OBJECT → CATEGORY CODES
    # -----------------------------------

    for column in synthetic_df.columns:

        if (
            pd.api.types.is_object_dtype(
                synthetic_df[column]
            )
            or str(
                synthetic_df[column].dtype
            ) == "category"
        ):

            synthetic_df[column] = (
                synthetic_df[column]
                .astype("category")
                .cat.codes
            )

    # -----------------------------------
    # BOOL → INT
    # -----------------------------------

    bool_columns = synthetic_df.select_dtypes(
        include=["bool"]
    ).columns

    for column in bool_columns:

        synthetic_df[column] = (
            synthetic_df[column]
            .astype(int)
        )

    # -----------------------------------
    # FORCE NUMERIC
    # -----------------------------------

    synthetic_df = (
        synthetic_df
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
    )

    # Placeholder target
    synthetic_df[target_column] = 0

    return synthetic_df


def generate_synthetic_dataset_from_reference(
    reference_df,
    feature_names,
    rows=1000
):
    synthetic_data = {}

    for feature in feature_names:

        if feature not in reference_df.columns:

            synthetic_data[feature] = np.random.randint(
                0,
                100,
                rows
            )

            continue

        column = reference_df[feature]

        # -----------------------------------
        # NUMERIC
        # -----------------------------------

        if pd.api.types.is_numeric_dtype(column):

            minimum = float(column.min())
            maximum = float(column.max())

            synthetic_data[feature] = np.random.uniform(
                minimum,
                maximum,
                rows
            )

        # -----------------------------------
        # CATEGORICAL
        # -----------------------------------

        else:

            values = (
                column
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if len(values) == 0:
                values = ["UNKNOWN"]

            synthetic_data[feature] = np.random.choice(
                values,
                rows
            )

    synthetic_df = pd.DataFrame(
        synthetic_data
    )

    # -----------------------------------
    # CLEAN TYPES
    # -----------------------------------

    for column in synthetic_df.columns:

        if (
            pd.api.types.is_object_dtype(
                synthetic_df[column]
            )
            or str(
                synthetic_df[column].dtype
            ) == "category"
        ):

            synthetic_df[column] = (
                synthetic_df[column]
                .astype("category")
                .cat.codes
            )

    bool_columns = synthetic_df.select_dtypes(
        include=["bool"]
    ).columns

    for column in bool_columns:

        synthetic_df[column] = (
            synthetic_df[column]
            .astype(int)
        )

    synthetic_df = (
        synthetic_df
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
        .fillna(0)
    )

    return synthetic_df


def generate_synthetic_target(
    model,
    X,
    task_type="classification",
    noise=0.15
):
    """
    Generates realistic targets
    with controlled noise.
    """

    try:

        features_df = X.copy()

        # -----------------------------------
        # CLEAN OBJECTS
        # -----------------------------------

        for column in features_df.columns:

            if (
                pd.api.types.is_object_dtype(
                    features_df[column]
                )
                or str(
                    features_df[column].dtype
                ) == "category"
            ):

                features_df[column] = (
                    features_df[column]
                    .astype("category")
                    .cat.codes
                )

        # -----------------------------------
        # BOOL → INT
        # -----------------------------------

        bool_columns = features_df.select_dtypes(
            include=["bool"]
        ).columns

        for column in bool_columns:

            features_df[column] = (
                features_df[column]
                .astype(int)
            )

        # -----------------------------------
        # FORCE NUMERIC
        # -----------------------------------

        features_df = (
            features_df
            .apply(
                pd.to_numeric,
                errors="coerce"
            )
            .fillna(0)
            .astype(float)
        )

        # -----------------------------------
        # MODEL PREDICTIONS
        # -----------------------------------

        predictions = model.predict(
            features_df
        )

        predictions = pd.Series(
            predictions
        ).copy()

        # -----------------------------------
        # ADD NOISE
        # -----------------------------------

        total_rows = len(predictions)

        noise_count = int(
            total_rows * noise
        )

        if noise_count > 0:

            noise_indices = np.random.choice(
                predictions.index,
                size=noise_count,
                replace=False
            )

            unique_classes = (
                predictions
                .unique()
                .tolist()
            )

            for index in noise_indices:

                current_value = (
                    predictions.loc[index]
                )

                possible_values = [

                    value

                    for value in unique_classes

                    if value != current_value
                ]

                if len(possible_values) > 0:

                    predictions.loc[index] = (
                        random.choice(
                            possible_values
                        )
                    )

        return predictions

    except Exception as error:

        raise Exception(
            f"Synthetic target generation failed: {str(error)}"
        )


def save_synthetic_dataset(
    df,
    output_path
):
    df.to_csv(
        output_path,
        index=False
    )

    return output_path