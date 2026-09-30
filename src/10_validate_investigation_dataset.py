import pandas as pd


DATA_PATH = "models/investigation_cases_enriched.csv"


def main():

    print("=" * 60)
    print("FRAUDLENS - INVESTIGATION DATASET VALIDATION")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    print(f"\nDataset shape: {df.shape}")

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("BASIC VALIDATION")
    print("=" * 60)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print(
        f"\nDuplicate case IDs: "
        f"{df['case_id'].duplicated().sum()}"
    )

    print(
        f"Duplicate transaction IDs: "
        f"{df['transaction_id'].duplicated().sum()}"
    )

    print(
        f"Missing case IDs: "
        f"{df['case_id'].isna().sum()}"
    )

    print(
        f"Missing transaction IDs: "
        f"{df['transaction_id'].isna().sum()}"
    )

    # --------------------------------------------------------
    # Risk distribution
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("RISK DISTRIBUTION")
    print("=" * 60)

    print(
        df["risk_level"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Investigation priority
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INVESTIGATION PRIORITY")
    print("=" * 60)

    print(
        df["investigation_priority"]
        .value_counts()
        .to_string()
    )

    # --------------------------------------------------------
    # Evidence count
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVIDENCE COUNT")
    print("=" * 60)

    print(
        df["evidence_count"]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # SHAP coverage
    # --------------------------------------------------------

    shap_columns = [
        "top_feature_1",
        "top_feature_2",
        "top_feature_3",
        "top_feature_4",
        "top_feature_5"
    ]

    print("\n" + "=" * 60)
    print("SHAP COVERAGE")
    print("=" * 60)

    for column in shap_columns:

        missing = df[column].isna().sum()

        print(
            f"{column}: "
            f"{len(df) - missing:,}/{len(df):,} "
            f"available"
        )

    # --------------------------------------------------------
    # Top SHAP features
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MOST FREQUENT TOP SHAP FEATURES")
    print("=" * 60)

    top_features = pd.concat(
        [
            df["top_feature_1"],
            df["top_feature_2"],
            df["top_feature_3"],
            df["top_feature_4"],
            df["top_feature_5"]
        ]
    )

    print(
        top_features
        .value_counts()
        .head(15)
        .to_string()
    )

    # --------------------------------------------------------
    # Actual fraud distribution
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("ACTUAL FRAUD LABEL")
    print("=" * 60)

    print(
        df["is_fraud"]
        .value_counts()
        .to_string()
    )

    print(
        f"\nActual fraud rate among "
        f"investigation cases: "
        f"{df['is_fraud'].mean():.2%}"
    )

    # --------------------------------------------------------
    # Probability summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("FRAUD PROBABILITY")
    print("=" * 60)

    print(
        df["fraud_probability"]
        .describe()
        .to_string()
    )

    # --------------------------------------------------------
    # Evidence examples
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SAMPLE INVESTIGATION CASES")
    print("=" * 60)

    sample_columns = [
        "case_id",
        "transaction_id",
        "fraud_probability",
        "risk_level",
        "investigation_priority",
        "evidence_count",
        "investigation_evidence",
        "top_feature_1",
        "top_feature_1_impact",
        "top_feature_2",
        "top_feature_2_impact",
        "is_fraud"
    ]

    print(
        df[sample_columns]
        .head(5)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    required_columns = [
        "case_id",
        "transaction_id",
        "account_id",
        "fraud_probability",
        "risk_level",
        "investigation_priority",
        "investigation_evidence",
        "evidence_count",
        "top_feature_1",
        "top_feature_1_shap",
        "top_feature_1_impact",
        "is_fraud"
    ]

    missing_required = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_required:

        raise ValueError(
            "Missing required columns: "
            + str(missing_required)
        )

    print("\n" + "=" * 60)
    print("FINAL VALIDATION")
    print("=" * 60)

    print("\nRequired columns: PASS")
    print("Dataset validation: PASS")

    print("\nInvestigation dataset is ready.")


if __name__ == "__main__":
    main()