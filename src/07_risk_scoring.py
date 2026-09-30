import json
import os

import pandas as pd
from xgboost import XGBClassifier


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/fraud_xgb.json"
THRESHOLD_PATH = "models/threshold.json"

TEST_MODEL_PATH = "data/processed/test_model.csv"
TEST_METADATA_PATH = "data/processed/test_metadata.csv"

OUTPUT_PATH = "models/risk_scored_transactions.csv"


# ============================================================
# LOAD THRESHOLD
# ============================================================

def load_threshold():

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    return config["threshold"]


# ============================================================
# CREATE RISK LEVEL
# ============================================================

def create_risk_level(probability):

    if probability >= 0.90:

        return "Critical"

    elif probability >= 0.70:

        return "High"

    elif probability >= 0.30:

        return "Medium"

    else:

        return "Low"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FRAUDLENS - RISK SCORING")
    print("=" * 60)

    # ========================================================
    # 1. CHECK REQUIRED FILES
    # ========================================================

    required_files = [
        MODEL_PATH,
        THRESHOLD_PATH,
        TEST_MODEL_PATH,
        TEST_METADATA_PATH
    ]

    for file_path in required_files:

        if not os.path.exists(file_path):

            raise FileNotFoundError(
                f"Required file not found: {file_path}"
            )

    # ========================================================
    # 2. LOAD MODEL
    # ========================================================

    print(
        "\nLoading trained XGBoost model..."
    )

    model = XGBClassifier()

    model.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    # ========================================================
    # 3. LOAD TEST DATA
    # ========================================================

    test_df = pd.read_csv(
        TEST_MODEL_PATH
    )

    metadata_df = pd.read_csv(
        TEST_METADATA_PATH,
        parse_dates=["timestamp"]
    )

    print(
        f"\nTest model data shape: "
        f"{test_df.shape}"
    )

    print(
        f"Test metadata shape: "
        f"{metadata_df.shape}"
    )

    # ========================================================
    # 4. VERIFY ROW ALIGNMENT
    # ========================================================

    if len(test_df) != len(metadata_df):

        raise ValueError(
            "Test model data and metadata "
            "have different row counts."
        )

    print(
        "\nMetadata alignment check: PASS"
    )

    # ========================================================
    # 5. PREPARE FEATURES
    # ========================================================

    X_test = test_df.drop(
        columns=["is_fraud"],
        errors="ignore"
    )

    # --------------------------------------------------------
    # Match categorical columns with training configuration
    # --------------------------------------------------------

    categorical_columns = [
        "merchant_category",
        "merchant_country",
        "device_type"
    ]

    for column in categorical_columns:

        if column in X_test.columns:

            X_test[column] = (
                X_test[column]
                .astype("category")
            )

    # ========================================================
    # 6. GENERATE FRAUD PROBABILITIES
    # ========================================================

    print(
        "\nGenerating fraud probabilities..."
    )

    fraud_probability = (
        model.predict_proba(X_test)[:, 1]
    )

    # ========================================================
    # 7. LOAD DECISION THRESHOLD
    # ========================================================

    threshold = load_threshold()

    print(
        f"\nFraud decision threshold: "
        f"{threshold:.2f}"
    )

    # ========================================================
    # 8. CREATE RISK DATAFRAME
    # ========================================================

    result = metadata_df.copy()

    result["fraud_probability"] = (
        fraud_probability
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    result["risk_level"] = (
        result["fraud_probability"]
        .apply(create_risk_level)
    )

    # --------------------------------------------------------
    # Fraud decision
    # --------------------------------------------------------

    result["decision"] = (
        result["fraud_probability"]
        >= threshold
    ).map(
        {
            True: "FRAUD_ALERT",
            False: "NORMAL"
        }
    )

    # --------------------------------------------------------
    # Add important model features
    # --------------------------------------------------------

    feature_columns = [
        "amount",
        "merchant_category",
        "merchant_country",
        "card_present",
        "device_type",
        "device_known",
        "ip_risk_score",
        "is_foreign_txn",
        "velocity_1h",
        "amount_vs_avg_ratio",
        "time_since_last_s",
        "account_age_days",
        "has_2fa",
        "credit_limit",
        "account_txn_count_before",
        "account_avg_amount_before",
        "credit_utilization",
        "amount_per_account_age",
        "foreign_high_risk",
        "velocity_amount_ratio",
        "hour_of_day",
        "day_of_week",
        "is_weekend"
    ]

    available_features = [
        column
        for column in feature_columns
        if column in test_df.columns
    ]

    for column in available_features:

        result[column] = test_df[column].values

    # --------------------------------------------------------
    # Actual fraud label
    # --------------------------------------------------------

    if "is_fraud" in test_df.columns:

        result["is_fraud"] = (
            test_df["is_fraud"]
            .values
        )

    # ========================================================
    # 9. CREATE CASE ID
    # ========================================================

    result.insert(
        0,
        "case_id",
        [
            f"FL-{i:06d}"
            for i in range(
                1,
                len(result) + 1
            )
        ]
    )

    # ========================================================
    # 10. REORDER COLUMNS
    # ========================================================

    preferred_columns = [
        "case_id",
        "transaction_id",
        "account_id",
        "timestamp",

        "amount",
        "fraud_probability",
        "risk_level",
        "decision",

        "merchant_category",
        "merchant_country",
        "card_present",
        "device_type",
        "device_known",

        "ip_risk_score",
        "is_foreign_txn",
        "velocity_1h",
        "amount_vs_avg_ratio",
        "time_since_last_s",

        "account_age_days",
        "account_txn_count_before",
        "account_avg_amount_before",

        "has_2fa",
        "credit_limit",

        "credit_utilization",
        "amount_per_account_age",
        "foreign_high_risk",
        "velocity_amount_ratio",

        "hour_of_day",
        "day_of_week",
        "is_weekend",

        "is_fraud"
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in result.columns
    ]

    result = result[
        available_columns
    ]

    # ========================================================
    # 11. VALIDATE TRANSACTION IDs
    # ========================================================

    if result[
        "transaction_id"
    ].isna().any():

        raise ValueError(
            "Missing transaction IDs detected."
        )

    if result[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs detected."
        )

    print(
        "\nTransaction ID validation: PASS"
    )

    # ========================================================
    # 12. RISK DISTRIBUTION
    # ========================================================

    print("\n" + "=" * 60)
    print("RISK DISTRIBUTION")
    print("=" * 60)

    risk_counts = (
        result[
            "risk_level"
        ]
        .value_counts()
    )

    risk_order = [
        "Critical",
        "High",
        "Medium",
        "Low"
    ]

    for risk in risk_order:

        count = risk_counts.get(
            risk,
            0
        )

        percentage = (
            count /
            len(result)
            * 100
        )

        print(
            f"{risk:<10} "
            f"{count:>8,} "
            f"({percentage:.2f}%)"
        )

    # ========================================================
    # 13. FRAUD ALERT SUMMARY
    # ========================================================

    fraud_alerts = (
        result["decision"]
        == "FRAUD_ALERT"
    ).sum()

    normal_transactions = (
        result["decision"]
        == "NORMAL"
    ).sum()

    print("\n" + "=" * 60)
    print("FRAUD DECISION SUMMARY")
    print("=" * 60)

    print(
        f"\nFraud Alerts: "
        f"{fraud_alerts:,}"
    )

    print(
        f"Normal Transactions: "
        f"{normal_transactions:,}"
    )

    # ========================================================
    # 14. TOP HIGH-RISK TRANSACTIONS
    # ========================================================

    print("\n" + "=" * 60)
    print("TOP 10 HIGHEST-RISK TRANSACTIONS")
    print("=" * 60)

    top_risk = (
        result
        .sort_values(
            "fraud_probability",
            ascending=False
        )
        .head(10)
    )

    print(
        top_risk[
            [
                "case_id",
                "transaction_id",
                "account_id",
                "amount",
                "fraud_probability",
                "risk_level",
                "decision"
            ]
        ].to_string(
            index=False
        )
    )

    # ========================================================
    # 15. RISK LEVEL VS ACTUAL FRAUD
    # ========================================================

    if "is_fraud" in result.columns:

        print("\n" + "=" * 60)
        print("RISK LEVEL VS ACTUAL FRAUD")
        print("=" * 60)

        risk_performance = (
            result
            .groupby("risk_level")
            .agg(
                transactions=(
                    "is_fraud",
                    "count"
                ),
                fraud_transactions=(
                    "is_fraud",
                    "sum"
                )
            )
        )

        risk_performance[
            "fraud_rate"
        ] = (
            risk_performance[
                "fraud_transactions"
            ]
            /
            risk_performance[
                "transactions"
            ]
            * 100
        )

        risk_performance = (
            risk_performance
            .reindex(risk_order)
        )

        print(
            risk_performance
            .to_string(
                formatters={
                    "fraud_rate":
                    "{:.2f}%".format
                }
            )
        )

    # ========================================================
    # 16. SAVE RESULT
    # ========================================================

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # ========================================================
    # 17. FINAL SUMMARY
    # ========================================================

    print("\n" + "=" * 60)
    print("RISK SCORING COMPLETE")
    print("=" * 60)

    print(
        f"\nOutput shape: "
        f"{result.shape}"
    )

    print(
        f"Saved:"
    )

    print(
        f"- {OUTPUT_PATH}"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()