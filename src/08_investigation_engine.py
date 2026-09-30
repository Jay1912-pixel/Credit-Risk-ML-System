import pandas as pd
import json


# ============================================================
# PATHS
# ============================================================

RISK_PATH = "models/risk_scored_transactions.csv"
MODEL_DATA_PATH = "data/processed/test_model.csv"
THRESHOLD_PATH = "models/threshold.json"

OUTPUT_PATH = "models/investigation_cases.csv"


# ============================================================
# LOAD THRESHOLD
# ============================================================

def load_threshold():

    with open(THRESHOLD_PATH, "r") as file:
        config = json.load(file)

    if "threshold" in config:
        return config["threshold"]

    if "fraud_threshold" in config:
        return config["fraud_threshold"]

    raise KeyError(
        "No threshold key found in threshold.json."
    )


# ============================================================
# GENERATE INVESTIGATION EVIDENCE
# ============================================================

def generate_evidence(row):

    evidence = []

    if row["amount_vs_avg_ratio"] >= 5:
        evidence.append(
            "Transaction amount is significantly higher "
            "than the account's average transaction amount."
        )

    if row["velocity_1h"] >= 5:
        evidence.append(
            "High transaction velocity detected "
            "within the previous one-hour period."
        )

    if row["ip_risk_score"] >= 50:
        evidence.append(
            "Elevated IP risk score detected."
        )

    if row["device_known"] == 0:
        evidence.append(
            "Transaction originated from an unknown device."
        )

    if row["is_foreign_txn"] == 1:
        evidence.append(
            "Transaction was identified as a foreign transaction."
        )

    if row["has_2fa"] == 0:
        evidence.append(
            "Transaction was completed without 2FA."
        )

    if row["card_present"] == 0:
        evidence.append(
            "Transaction was not card-present."
        )

    if row["account_txn_count_before"] == 0:
        evidence.append(
            "This is the first observed transaction "
            "for the account."
        )

    if row["credit_utilization"] >= 1:
        evidence.append(
            "Transaction amount is at or above "
            "the account credit limit."
        )

    if row["amount_per_account_age"] > 100:
        evidence.append(
            "High transaction amount relative "
            "to account age."
        )

    if not evidence:
        evidence.append(
            "No major rule-based risk signal detected."
        )

    return evidence


# ============================================================
# INVESTIGATION PRIORITY
# ============================================================

def assign_priority(risk_level):

    priority_map = {
        "Critical": "Immediate",
        "High": "High",
        "Medium": "Review",
        "Low": "Low"
    }

    return priority_map.get(
        risk_level,
        "Review"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FRAUDLENS - INVESTIGATION ENGINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Load risk scoring data
    # --------------------------------------------------------

    print("\nLoading risk scoring data...")

    risk_df = pd.read_csv(
        RISK_PATH
    )

    print(
        f"Risk scoring data shape: "
        f"{risk_df.shape}"
    )

    print("\nRisk scoring columns:")

    for column in risk_df.columns:
        print(f"- {column}")

    # --------------------------------------------------------
    # Validate required metadata
    # --------------------------------------------------------

    required_metadata = [
        "case_id",
        "transaction_id",
        "account_id",
        "timestamp",
        "fraud_probability",
        "risk_level",
        "decision"
    ]

    missing_metadata = [
        column
        for column in required_metadata
        if column not in risk_df.columns
    ]

    if missing_metadata:

        raise ValueError(
            "Missing required columns in risk scoring data: "
            + ", ".join(missing_metadata)
        )

    # --------------------------------------------------------
    # Load model data
    # --------------------------------------------------------

    print("\nLoading model data...")

    model_df = pd.read_csv(
        MODEL_DATA_PATH
    )

    print(
        f"Model data shape: "
        f"{model_df.shape}"
    )

    # --------------------------------------------------------
    # Row alignment validation
    # --------------------------------------------------------

    if len(risk_df) != len(model_df):

        raise ValueError(
            "Risk scoring data and model data "
            "row counts do not match."
        )

    print(
        "\nRow alignment check: PASS"
    )

    print(
        f"Rows available for investigation: "
        f"{len(risk_df):,}"
    )

    # --------------------------------------------------------
    # Threshold
    # --------------------------------------------------------

    threshold = load_threshold()

    print(
        f"\nFraud decision threshold: "
        f"{threshold:.2f}"
    )

    # --------------------------------------------------------
    # Identify fraud alerts
    # --------------------------------------------------------

    investigation_mask = (
        risk_df["decision"] == "FRAUD_ALERT"
    )

    investigation_df = risk_df[
        investigation_mask
    ].copy()

    print(
        f"\nInvestigation cases: "
        f"{len(investigation_df):,}"
    )

    if len(investigation_df) == 0:

        print(
            "\nNo fraud alerts found."
        )

        return

    # --------------------------------------------------------
    # Generate evidence
    # --------------------------------------------------------

    print(
        "\nGenerating investigation evidence..."
    )

    evidence_results = []

    for _, row in investigation_df.iterrows():

        evidence = generate_evidence(
            row
        )

        evidence_results.append(
            evidence
        )

    investigation_df[
        "investigation_evidence"
    ] = [
        " | ".join(evidence)
        for evidence in evidence_results
    ]

    investigation_df[
        "evidence_count"
    ] = [
        len(evidence)
        for evidence in evidence_results
    ]

    # --------------------------------------------------------
    # Investigation priority
    # --------------------------------------------------------

    investigation_df[
        "investigation_priority"
    ] = investigation_df[
        "risk_level"
    ].apply(assign_priority)

    # --------------------------------------------------------
    # Keep actual fraud label
    # --------------------------------------------------------

    if "is_fraud" not in investigation_df.columns:

        investigation_df["is_fraud"] = (
            model_df.loc[
                investigation_df.index,
                "is_fraud"
            ].values
        )

    # --------------------------------------------------------
    # Column ordering
    # --------------------------------------------------------

    preferred_columns = [
        "case_id",
        "transaction_id",
        "account_id",
        "timestamp",
        "amount",
        "fraud_probability",
        "risk_level",
        "decision",
        "investigation_priority",
        "evidence_count",
        "investigation_evidence",
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

    final_columns = [
        column
        for column in preferred_columns
        if column in investigation_df.columns
    ]

    investigation_df = investigation_df[
        final_columns
    ].copy()

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if investigation_df[
        "transaction_id"
    ].isnull().any():

        raise ValueError(
            "Null transaction IDs found."
        )

    if investigation_df[
        "transaction_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate transaction IDs found."
        )

    if investigation_df[
        "account_id"
    ].isnull().any():

        raise ValueError(
            "Null account IDs found."
        )

    print(
        "\nTransaction metadata validation: PASS"
    )

    # --------------------------------------------------------
    # TOP CASES
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TOP INVESTIGATION CASES")
    print("=" * 60)

    top_columns = [
        "case_id",
        "transaction_id",
        "account_id",
        "amount",
        "fraud_probability",
        "risk_level",
        "investigation_priority",
        "evidence_count",
        "is_fraud"
    ]

    print(
        investigation_df[
            top_columns
        ]
        .sort_values(
            "fraud_probability",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # SAMPLE CASE
    # --------------------------------------------------------

    sample_case = (
        investigation_df
        .sort_values(
            "fraud_probability",
            ascending=False
        )
        .iloc[0]
    )

    print("\n" + "=" * 60)
    print("SAMPLE INVESTIGATION CASE")
    print("=" * 60)

    print(
        f"\nCase ID: "
        f"{sample_case['case_id']}"
    )

    print(
        f"Transaction ID: "
        f"{sample_case['transaction_id']}"
    )

    print(
        f"Account ID: "
        f"{sample_case['account_id']}"
    )

    print(
        f"Timestamp: "
        f"{sample_case['timestamp']}"
    )

    print(
        f"Fraud Probability: "
        f"{sample_case['fraud_probability']:.6f}"
    )

    print(
        f"Risk Level: "
        f"{sample_case['risk_level']}"
    )

    print(
        f"Investigation Priority: "
        f"{sample_case['investigation_priority']}"
    )

    print(
        f"Transaction Amount: "
        f"{sample_case['amount']:.2f}"
    )

    print(
        f"IP Risk Score: "
        f"{sample_case['ip_risk_score']:.2f}"
    )

    print(
        f"Velocity 1H: "
        f"{sample_case['velocity_1h']}"
    )

    print(
        f"Amount vs Average Ratio: "
        f"{sample_case['amount_vs_avg_ratio']:.2f}"
    )

    print(
        f"Foreign Transaction: "
        f"{sample_case['is_foreign_txn']}"
    )

    print(
        f"Known Device: "
        f"{sample_case['device_known']}"
    )

    print(
        f"2FA Used: "
        f"{sample_case['has_2fa']}"
    )

    print("\nEvidence:")

    for evidence in (
        sample_case[
            "investigation_evidence"
        ].split(" | ")
    ):

        print(
            f"- {evidence}"
        )

    # --------------------------------------------------------
    # PRIORITY DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INVESTIGATION PRIORITY DISTRIBUTION")
    print("=" * 60)

    print(
        investigation_df[
            "investigation_priority"
        ].value_counts()
    )

    # --------------------------------------------------------
    # EVIDENCE DISTRIBUTION
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVIDENCE COUNT DISTRIBUTION")
    print("=" * 60)

    print(
        investigation_df[
            "evidence_count"
        ].value_counts().sort_index()
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    investigation_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("INVESTIGATION ENGINE COMPLETE")
    print("=" * 60)

    print(
        f"\nInvestigation cases generated: "
        f"{len(investigation_df):,}"
    )

    print(
        "\nOutput file:"
    )

    print(
        f"- {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()