import pandas as pd
import numpy as np
import os


# ============================================================
# PATHS
# ============================================================

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"

TRAIN_MODEL_PATH = "data/processed/train_model.csv"
TEST_MODEL_PATH = "data/processed/test_model.csv"

TRAIN_METADATA_PATH = "data/processed/train_metadata.csv"
TEST_METADATA_PATH = "data/processed/test_metadata.csv"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # Sort by account and time
    # --------------------------------------------------------

    df = df.sort_values(
        ["account_id", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    df["transaction_year"] = df["timestamp"].dt.year

    df["transaction_month"] = df["timestamp"].dt.month

    df["transaction_day"] = df["timestamp"].dt.day

    df["transaction_hour"] = df["timestamp"].dt.hour

    # --------------------------------------------------------
    # Derived transaction features
    # --------------------------------------------------------

    df["credit_utilization"] = (
        df["amount"] /
        df["credit_limit"]
    )

    df["amount_per_account_age"] = (
        df["amount"] /
        (df["account_age_days"] + 1)
    )

    df["foreign_high_risk"] = (
        df["is_foreign_txn"] *
        df["ip_risk_score"]
    )

    df["velocity_amount_ratio"] = (
        df["velocity_1h"] *
        df["amount_vs_avg_ratio"]
    )

    # --------------------------------------------------------
    # Point-in-time account transaction count
    # --------------------------------------------------------

    df["account_txn_count_before"] = (
        df.groupby("account_id")
        .cumcount()
    )

    # --------------------------------------------------------
    # Point-in-time account average amount
    # --------------------------------------------------------

    df["account_avg_amount_before"] = (
        df.groupby("account_id")["amount"]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .mean()
        )
        .fillna(0)
    )

    return df


# ============================================================
# PREPARE MODEL DATA
# ============================================================

def prepare_model_data(df):

    model_df = df.copy()

    # --------------------------------------------------------
    # Columns not used by the model
    # --------------------------------------------------------

    columns_to_drop = [
        "transaction_id",
        "account_id",
        "timestamp",
        "fraud_pattern"
    ]

    model_df = model_df.drop(
        columns=columns_to_drop,
        errors="ignore"
    )

    return model_df


# ============================================================
# VALIDATION
# ============================================================

def validate_data(train_df, test_df):

    print("\n" + "=" * 60)
    print("DATA VALIDATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    print("\nMissing Values:")

    missing_train = (
        train_df.isnull()
        .sum()
    )

    missing_test = (
        test_df.isnull()
        .sum()
    )

    print(
        "\nTrain:"
    )

    print(
        missing_train[
            missing_train > 0
        ]
    )

    print(
        "\nTest:"
    )

    print(
        missing_test[
            missing_test > 0
        ]
    )

    # --------------------------------------------------------
    # Infinite values
    # --------------------------------------------------------

    numeric_train = train_df.select_dtypes(
        include=np.number
    )

    numeric_test = test_df.select_dtypes(
        include=np.number
    )

    train_inf = np.isinf(
        numeric_train
    ).sum().sum()

    test_inf = np.isinf(
        numeric_test
    ).sum().sum()

    print(
        f"\nInfinite values in train: "
        f"{train_inf}"
    )

    print(
        f"Infinite values in test: "
        f"{test_inf}"
    )

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------

    print(
        "\nTarget Distribution:"
    )

    print(
        train_df["is_fraud"]
        .value_counts()
    )

    print(
        "\nFraud Rate:"
    )

    print(
        f"{train_df['is_fraud'].mean():.2%}"
    )

    # --------------------------------------------------------
    # Account feature validation
    # --------------------------------------------------------

    first_transaction_count = (
        train_df[
            "account_txn_count_before"
        ] == 0
    ).sum()

    first_average_count = (
        train_df[
            "account_avg_amount_before"
        ] == 0
    ).sum()

    print(
        "\nPoint-in-Time Account Feature Validation:"
    )

    print(
        f"Train rows with "
        f"account_txn_count_before = 0: "
        f"{first_transaction_count:,}"
    )

    print(
        f"Train rows with "
        f"account_avg_amount_before = 0: "
        f"{first_average_count:,}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FRAUDLENS - MODEL DATA PREPARATION")
    print("=" * 60)

    # ========================================================
    # 1. LOAD DATA
    # ========================================================

    train_raw = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["timestamp"]
    )

    test_raw = pd.read_csv(
        TEST_PATH,
        parse_dates=["timestamp"]
    )

    print(
        f"\nOriginal train shape: "
        f"{train_raw.shape}"
    )

    print(
        f"Original test shape: "
        f"{test_raw.shape}"
    )

    # ========================================================
    # 2. COMBINE DATA
    # ========================================================

    combined = pd.concat(
        [
            train_raw,
            test_raw
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Preserve original train/test boundary
    # --------------------------------------------------------

    train_end_timestamp = (
        train_raw["timestamp"].max()
    )

    # ========================================================
    # 3. CREATE POINT-IN-TIME FEATURES
    # ========================================================

    print(
        "\nCreating point-in-time features..."
    )

    combined = create_features(
        combined
    )

    # ========================================================
    # 4. RESTORE TRAIN / TEST SPLIT
    # ========================================================

    train_featured = combined[
        combined["timestamp"]
        <= train_end_timestamp
    ].copy()

    test_featured = combined[
        combined["timestamp"]
        > train_end_timestamp
    ].copy()

    # ========================================================
    # 5. CREATE METADATA
    # ========================================================

    # These columns are NOT used by the ML model.
    # They are preserved for investigation and traceability.

    metadata_columns = [
        "transaction_id",
        "account_id",
        "timestamp"
    ]

    train_metadata = train_featured[
        metadata_columns
    ].copy()

    test_metadata = test_featured[
        metadata_columns
    ].copy()

    # --------------------------------------------------------
    # Add stable case row identifier
    # --------------------------------------------------------

    train_metadata.insert(
        0,
        "metadata_index",
        range(
            len(train_metadata)
        )
    )

    test_metadata.insert(
        0,
        "metadata_index",
        range(
            len(test_metadata)
        )
    )

    # ========================================================
    # 6. PREPARE MODEL DATA
    # ========================================================

    train_model = prepare_model_data(
        train_featured
    )

    test_model = prepare_model_data(
        test_featured
    )

    # ========================================================
    # 7. RESET INDEX
    # ========================================================

    train_model = (
        train_model
        .reset_index(drop=True)
    )

    test_model = (
        test_model
        .reset_index(drop=True)
    )

    train_metadata = (
        train_metadata
        .reset_index(drop=True)
    )

    test_metadata = (
        test_metadata
        .reset_index(drop=True)
    )

    # ========================================================
    # 8. VERIFY METADATA ALIGNMENT
    # ========================================================

    if len(train_model) != len(train_metadata):

        raise ValueError(
            "Train model data and metadata "
            "row counts do not match."
        )

    if len(test_model) != len(test_metadata):

        raise ValueError(
            "Test model data and metadata "
            "row counts do not match."
        )

    # --------------------------------------------------------
    # Verify transaction IDs are unique
    # --------------------------------------------------------

    if train_metadata[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs found "
            "in train metadata."
        )

    if test_metadata[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs found "
            "in test metadata."
        )

    print(
        "\nMetadata alignment check: PASS"
    )

    # ========================================================
    # 9. VALIDATE MODEL DATA
    # ========================================================

    validate_data(
        train_model,
        test_model
    )

    # ========================================================
    # 10. SAVE MODEL DATA
    # ========================================================

    train_model.to_csv(
        TRAIN_MODEL_PATH,
        index=False
    )

    test_model.to_csv(
        TEST_MODEL_PATH,
        index=False
    )

    # ========================================================
    # 11. SAVE METADATA
    # ========================================================

    train_metadata.to_csv(
        TRAIN_METADATA_PATH,
        index=False
    )

    test_metadata.to_csv(
        TEST_METADATA_PATH,
        index=False
    )

    # ========================================================
    # 12. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("MODEL DATA PREPARATION COMPLETE")
    print("=" * 60)

    print(
        f"\nTrain model shape: "
        f"{train_model.shape}"
    )

    print(
        f"Test model shape: "
        f"{test_model.shape}"
    )

    print(
        f"\nTrain metadata shape: "
        f"{train_metadata.shape}"
    )

    print(
        f"Test metadata shape: "
        f"{test_metadata.shape}"
    )

    print(
        "\nFiles saved:"
    )

    print(
        f"- {TRAIN_MODEL_PATH}"
    )

    print(
        f"- {TEST_MODEL_PATH}"
    )

    print(
        f"- {TRAIN_METADATA_PATH}"
    )

    print(
        f"- {TEST_METADATA_PATH}"
    )


if __name__ == "__main__":

    main()