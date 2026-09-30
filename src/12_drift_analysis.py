import os
import pandas as pd
import numpy as np


# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "train_model.csv"
)

TEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "test_model.csv"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "models")

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "data_drift_report.csv"
)


# ============================================================
# Load data
# ============================================================

print("Loading training data...")
train_df = pd.read_csv(TRAIN_PATH)

print("Loading test data...")
test_df = pd.read_csv(TEST_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")


# ============================================================
# Features to monitor
# ============================================================

numeric_features = [
    "amount",
    "ip_risk_score",
    "velocity_1h",
    "amount_vs_avg_ratio",
    "time_since_last_s",
    "account_age_days",
    "credit_limit",
    "credit_utilization",
    "account_txn_count_before",
    "amount_per_account_age",
    "velocity_amount_ratio",
]

categorical_features = [
    "merchant_category",
    "merchant_country",
    "device_type",
]


# ============================================================
# Numeric drift
# ============================================================

def calculate_numeric_drift(train_series, test_series):
    train_series = pd.to_numeric(train_series, errors="coerce").dropna()
    test_series = pd.to_numeric(test_series, errors="coerce").dropna()

    train_mean = train_series.mean()
    test_mean = test_series.mean()

    train_median = train_series.median()
    test_median = test_series.median()

    train_std = train_series.std()
    test_std = test_series.std()

    mean_change_pct = (
        ((test_mean - train_mean) / abs(train_mean)) * 100
        if train_mean != 0
        else np.nan
    )

    return {
        "feature": train_series.name,
        "feature_type": "numeric",
        "train_mean": train_mean,
        "test_mean": test_mean,
        "train_median": train_median,
        "test_median": test_median,
        "train_std": train_std,
        "test_std": test_std,
        "mean_change_pct": mean_change_pct,
    }


numeric_results = []

for feature in numeric_features:

    if feature not in train_df.columns or feature not in test_df.columns:
        print(f"Skipping missing feature: {feature}")
        continue

    result = calculate_numeric_drift(
        train_df[feature],
        test_df[feature]
    )

    numeric_results.append(result)


# ============================================================
# Categorical drift
# ============================================================

def calculate_categorical_drift(train_series, test_series):

    train_distribution = train_series.value_counts(
        normalize=True
    )

    test_distribution = test_series.value_counts(
        normalize=True
    )

    categories = set(train_distribution.index).union(
        set(test_distribution.index)
    )

    max_distribution_change = 0

    for category in categories:

        train_pct = train_distribution.get(category, 0)
        test_pct = test_distribution.get(category, 0)

        change = abs(test_pct - train_pct)

        max_distribution_change = max(
            max_distribution_change,
            change
        )

    return {
        "feature": train_series.name,
        "feature_type": "categorical",
        "train_mean": np.nan,
        "test_mean": np.nan,
        "train_median": np.nan,
        "test_median": np.nan,
        "train_std": np.nan,
        "test_std": np.nan,
        "mean_change_pct": max_distribution_change * 100,
    }


categorical_results = []

for feature in categorical_features:

    if feature not in train_df.columns or feature not in test_df.columns:
        print(f"Skipping missing feature: {feature}")
        continue

    result = calculate_categorical_drift(
        train_df[feature],
        test_df[feature]
    )

    categorical_results.append(result)


# ============================================================
# Combine results
# ============================================================

results = numeric_results + categorical_results

drift_report = pd.DataFrame(results)


# ============================================================
# Add drift status
# ============================================================

def classify_drift(change):

    if pd.isna(change):
        return "UNKNOWN"

    change = abs(change)

    if change < 10:
        return "LOW"

    elif change < 20:
        return "MEDIUM"

    else:
        return "HIGH"


drift_report["drift_status"] = (
    drift_report["mean_change_pct"]
    .apply(classify_drift)
)


# ============================================================
# Save report
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

drift_report.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# Print summary
# ============================================================

print("\n" + "=" * 60)
print("DATA DRIFT ANALYSIS")
print("=" * 60)

print(
    drift_report[
        [
            "feature",
            "feature_type",
            "mean_change_pct",
            "drift_status"
        ]
    ].to_string(index=False)
)

print("\nDrift summary:")

print(
    drift_report["drift_status"]
    .value_counts()
)

print(f"\nReport saved to:")
print(OUTPUT_PATH)

print("\nDrift analysis completed successfully.")