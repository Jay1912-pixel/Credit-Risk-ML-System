import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import json


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/fraud_xgb.json"
THRESHOLD_PATH = "models/threshold.json"

TEST_MODEL_PATH = "data/processed/test_model.csv"
RISK_PATH = "models/risk_scored_transactions.csv"
INVESTIGATION_PATH = "models/investigation_cases.csv"

GLOBAL_SHAP_PATH = "models/shap_feature_importance.csv"
SAMPLE_EXPLANATION_PATH = "models/sample_transaction_explanation.csv"
CASE_SHAP_PATH = "models/investigation_case_explanations.csv"


# ============================================================
# CONFIGURATION
# ============================================================

CATEGORICAL_COLUMNS = [
    "merchant_category",
    "merchant_country",
    "device_type"
]

SHAP_SAMPLE_SIZE = 2000
TOP_N_FEATURES = 5


# ============================================================
# LOAD THRESHOLD
# ============================================================

def load_threshold():

    with open(THRESHOLD_PATH, "r") as file:
        config = json.load(file)

    # Support both possible key names
    if "threshold" in config:
        return config["threshold"]

    if "fraud_threshold" in config:
        return config["fraud_threshold"]

    raise KeyError(
        "No threshold key found in threshold.json. "
        "Expected 'threshold' or 'fraud_threshold'."
    )


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(df):

    df = df.copy()

    for column in CATEGORICAL_COLUMNS:

        if column in df.columns:
            df[column] = df[column].astype("category")

    return df


# ============================================================
# GET MODEL FEATURES
# ============================================================

def get_model_features(df):

    return df.drop(
        columns=["is_fraud"],
        errors="ignore"
    )


# ============================================================
# CALCULATE SHAP VALUES
# ============================================================

def calculate_shap_values(model, X):

    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # Handle binary classification output
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print("SHAP values calculated successfully.")

    return explainer, shap_values


# ============================================================
# GLOBAL SHAP FEATURE IMPORTANCE
# ============================================================

def save_global_importance(X, shap_values):

    importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_absolute_shap": importance
    })

    importance_df = importance_df.sort_values(
        "mean_absolute_shap",
        ascending=False
    ).reset_index(drop=True)

    importance_df.to_csv(
        GLOBAL_SHAP_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("GLOBAL SHAP FEATURE IMPORTANCE")
    print("=" * 60)

    print("\nTop 15 Features:")

    for _, row in importance_df.head(15).iterrows():

        print(
            f"{row['feature']:<30}"
            f"{row['mean_absolute_shap']:.6f}"
        )

    print("\nSaved global SHAP importance:")
    print(f"- {GLOBAL_SHAP_PATH}")

    return importance_df


# ============================================================
# SINGLE TRANSACTION EXPLANATION
# ============================================================

def save_sample_explanation(
    X_sample,
    shap_values,
    risk_df,
    threshold
):

    sample_index = 0

    transaction_id = (
        risk_df.iloc[sample_index]["transaction_id"]
    )

    fraud_probability = (
        risk_df.iloc[sample_index]["fraud_probability"]
    )

    actual_label = (
        risk_df.iloc[sample_index]["is_fraud"]
    )

    row_shap = shap_values[sample_index]

    contribution_df = pd.DataFrame({
        "feature": X_sample.columns,
        "shap_value": row_shap
    })

    contribution_df["absolute_shap"] = (
        contribution_df["shap_value"].abs()
    )

    contribution_df["impact"] = np.where(
        contribution_df["shap_value"] > 0,
        "increased fraud risk",
        "decreased fraud risk"
    )

    contribution_df = contribution_df.sort_values(
        "absolute_shap",
        ascending=False
    ).head(10)

    contribution_df.insert(
        0,
        "transaction_id",
        transaction_id
    )

    contribution_df.insert(
        1,
        "fraud_probability",
        fraud_probability
    )

    contribution_df.insert(
        2,
        "threshold",
        threshold
    )

    contribution_df.insert(
        3,
        "actual_fraud",
        actual_label
    )

    contribution_df.to_csv(
        SAMPLE_EXPLANATION_PATH,
        index=False
    )

    print("\n" + "=" * 60)
    print("SINGLE TRANSACTION EXPLANATION")
    print("=" * 60)

    print(
        f"\nTransaction ID: {transaction_id}"
    )

    print(
        f"Fraud Probability: "
        f"{fraud_probability:.4f}"
    )

    print(
        f"Decision Threshold: "
        f"{threshold:.2f}"
    )

    decision = (
        "FRAUD_ALERT"
        if fraud_probability >= threshold
        else "NORMAL"
    )

    print(
        f"Decision: {decision}"
    )

    print(
        f"Actual Label: {actual_label}"
    )

    print("\nTop Contributing Features:")

    for _, row in contribution_df.iterrows():

        print(
            f"- {row['feature']}: "
            f"SHAP={row['shap_value']:.6f} "
            f"({row['impact']})"
        )

    print("\nSaved transaction explanation:")
    print(
        f"- {SAMPLE_EXPLANATION_PATH}"
    )


# ============================================================
# GENERATE CASE-LEVEL SHAP EXPLANATIONS
# ============================================================

def generate_case_explanations(
    model,
    investigation_df,
    test_df,
    risk_df
):

    print("\n" + "=" * 60)
    print("INVESTIGATION CASE SHAP EXPLANATIONS")
    print("=" * 60)

    print(
        f"\nInvestigation cases: "
        f"{len(investigation_df):,}"
    )

    # --------------------------------------------------------
    # Match investigation transactions to test rows
    # --------------------------------------------------------

    transaction_to_index = pd.Series(
        test_df.index,
        index=risk_df["transaction_id"]
    )

    investigation_indices = (
        investigation_df["transaction_id"]
        .map(transaction_to_index)
    )

    if investigation_indices.isnull().any():

        missing_count = (
            investigation_indices.isnull().sum()
        )

        raise ValueError(
            f"{missing_count} investigation transactions "
            "could not be matched to test model data."
        )

    investigation_indices = (
        investigation_indices.astype(int)
    )

    # --------------------------------------------------------
    # Get exact model rows
    # --------------------------------------------------------

    investigation_model_df = test_df.loc[
        investigation_indices
    ].copy()

    investigation_model_df.reset_index(
        drop=True,
        inplace=True
    )

    investigation_df = (
        investigation_df.reset_index(drop=True)
    )

    X_cases = get_model_features(
        investigation_model_df
    )

    X_cases = prepare_features(
        X_cases
    )

    # --------------------------------------------------------
    # Calculate SHAP
    # --------------------------------------------------------

    print(
        "\nCalculating SHAP values for "
        "investigation cases..."
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_cases
    )

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    print(
        "Investigation case SHAP values "
        "calculated successfully."
    )

    # --------------------------------------------------------
    # Create output records
    # --------------------------------------------------------

    records = []

    for i in range(
        len(investigation_df)
    ):

        case = investigation_df.iloc[i]

        row_shap = shap_values[i]

        contribution_df = pd.DataFrame({
            "feature": X_cases.columns,
            "shap_value": row_shap
        })

        contribution_df["absolute_shap"] = (
            contribution_df["shap_value"].abs()
        )

        contribution_df = contribution_df.sort_values(
            "absolute_shap",
            ascending=False
        ).head(TOP_N_FEATURES)

        record = {
            "case_id": case["case_id"],
            "transaction_id": case["transaction_id"],
            "account_id": case["account_id"],
            "timestamp": case["timestamp"],
            "fraud_probability": case[
                "fraud_probability"
            ],
            "risk_level": case[
                "risk_level"
            ],
            "investigation_priority": case[
                "investigation_priority"
            ]
        }

        for rank, (_, contribution) in enumerate(
            contribution_df.iterrows(),
            start=1
        ):

            record[
                f"top_feature_{rank}"
            ] = contribution["feature"]

            record[
                f"top_feature_{rank}_shap"
            ] = contribution["shap_value"]

            record[
                f"top_feature_{rank}_impact"
            ] = (
                "increased fraud risk"
                if contribution["shap_value"] > 0
                else "decreased fraud risk"
            )

        records.append(record)

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    case_explanation_df = pd.DataFrame(
        records
    )

    case_explanation_df.to_csv(
        CASE_SHAP_PATH,
        index=False
    )

    print(
        f"\nGenerated explanations for "
        f"{len(case_explanation_df):,} "
        "investigation cases."
    )

    print(
        "\nSaved case-level SHAP explanations:"
    )

    print(
        f"- {CASE_SHAP_PATH}"
    )

    return case_explanation_df


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FRAUDLENS - SHAP EXPLAINABILITY")
    print("=" * 60)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading trained model...")

    model = xgb.XGBClassifier()

    model.load_model(
        MODEL_PATH
    )

    print(
        "Model loaded successfully."
    )

    print(
        f"- {MODEL_PATH}"
    )

    # --------------------------------------------------------
    # Load threshold
    # --------------------------------------------------------

    threshold = load_threshold()

    print(
        f"\nFraud Threshold: "
        f"{threshold:.2f}"
    )

    # --------------------------------------------------------
    # Load test model data
    # --------------------------------------------------------

    print(
        "\nLoading test model data..."
    )

    test_df = pd.read_csv(
        TEST_MODEL_PATH
    )

    print(
        f"Test Data Shape: "
        f"{test_df.shape}"
    )

    X = get_model_features(
        test_df
    )

    X = prepare_features(
        X
    )

    # --------------------------------------------------------
    # Categorical columns
    # --------------------------------------------------------

    print(
        "\nCategorical Columns:"
    )

    for column in CATEGORICAL_COLUMNS:
        print(
            f"- {column}"
        )

    # --------------------------------------------------------
    # GLOBAL SHAP
    # --------------------------------------------------------

    sample_size = min(
        SHAP_SAMPLE_SIZE,
        len(X)
    )

    X_sample = X.iloc[
        :sample_size
    ].copy()

    print(
        f"\nSHAP Sample Size: "
        f"{len(X_sample):,}"
    )

    explainer, shap_values = (
        calculate_shap_values(
            model,
            X_sample
        )
    )

    save_global_importance(
        X_sample,
        shap_values
    )

    # --------------------------------------------------------
    # LOAD RISK DATA
    # --------------------------------------------------------

    print(
        "\nLoading risk scoring data..."
    )

    risk_df = pd.read_csv(
        RISK_PATH
    )

    print(
        f"Risk Data Shape: "
        f"{risk_df.shape}"
    )

    if len(risk_df) != len(test_df):

        raise ValueError(
            "Risk scoring data and test model "
            "row counts do not match."
        )

    print(
        "Risk/test row alignment: PASS"
    )

    # --------------------------------------------------------
    # SINGLE TRANSACTION EXPLANATION
    # --------------------------------------------------------

    save_sample_explanation(
        X_sample,
        shap_values,
        risk_df,
        threshold
    )

    # --------------------------------------------------------
    # LOAD INVESTIGATION CASES
    # --------------------------------------------------------

    print(
        "\nLoading investigation cases..."
    )

    investigation_df = pd.read_csv(
        INVESTIGATION_PATH
    )

    print(
        f"Investigation Data Shape: "
        f"{investigation_df.shape}"
    )

    if len(investigation_df) == 0:

        raise ValueError(
            "No investigation cases found."
        )

    # --------------------------------------------------------
    # CASE-LEVEL SHAP
    # --------------------------------------------------------

    generate_case_explanations(
        model,
        investigation_df,
        test_df,
        risk_df
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("SHAP EXPLAINABILITY COMPLETE")
    print("=" * 60)

    print(
        "\nGenerated Artifacts:"
    )

    print(
        f"- {GLOBAL_SHAP_PATH}"
    )

    print(
        f"- {SAMPLE_EXPLANATION_PATH}"
    )

    print(
        f"- {CASE_SHAP_PATH}"
    )


if __name__ == "__main__":
    main()