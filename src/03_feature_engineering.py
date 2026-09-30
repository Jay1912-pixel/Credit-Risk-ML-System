import pandas as pd

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


def create_features(df):
    df = df.copy()

    # Time-based features
    df["transaction_year"] = df["timestamp"].dt.year
    df["transaction_month"] = df["timestamp"].dt.month
    df["transaction_day"] = df["timestamp"].dt.day
    df["transaction_hour"] = df["timestamp"].dt.hour

    # Financial behavior
    df["credit_utilization"] = df["amount"] / df["credit_limit"]
    df["amount_per_account_age"] = (
        df["amount"] / (df["account_age_days"] + 1)
    )

    # Risk interaction features
    df["foreign_high_risk"] = (
        df["is_foreign_txn"] * df["ip_risk_score"]
    )

    df["velocity_amount_ratio"] = (
        df["velocity_1h"] * df["amount_vs_avg_ratio"]
    )

    return df


def inspect_account_behavior(df):
    account_counts = df["account_id"].value_counts()

    print("\n" + "=" * 60)
    print("ACCOUNT BEHAVIOR")
    print("=" * 60)

    print(f"\nUnique Accounts: {account_counts.size:,}")
    print(
        f"Average Transactions per Account: "
        f"{account_counts.mean():.2f}"
    )
    print(
        f"Median Transactions per Account: "
        f"{account_counts.median():.0f}"
    )
    print(
        f"Maximum Transactions for One Account: "
        f"{account_counts.max():,}"
    )

    print("\nTransactions per Account:")
    print(account_counts.describe())


def inspect_account_fraud_behavior(df):
    account_fraud = (
        df.groupby("account_id")
        .agg(
            transaction_count=("transaction_id", "count"),
            fraud_count=("is_fraud", "sum"),
            fraud_rate=("is_fraud", "mean")
        )
        .sort_values("fraud_count", ascending=False)
    )

    print("\n" + "=" * 60)
    print("ACCOUNT FRAUD BEHAVIOR")
    print("=" * 60)

    print("\nAccounts with the Most Fraud Transactions:")
    print(account_fraud.head(10))

    print("\nAccounts with At Least One Fraud:")
    print((account_fraud["fraud_count"] > 0).sum())

    print("\nMaximum Fraud Transactions for One Account:")
    print(account_fraud["fraud_count"].max())


def main():
    train_df = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["timestamp"]
    )

    test_df = pd.read_csv(
        TEST_PATH,
        parse_dates=["timestamp"]
    )

    train_df = create_features(train_df)
    test_df = create_features(test_df)

    print("=" * 60)
    print("FRAUDLENS - FEATURE ENGINEERING")
    print("=" * 60)

    print(f"\nOriginal feature count: 23")
    print(f"New training shape: {train_df.shape}")
    print(f"New test shape: {test_df.shape}")

    print("\nNew Features:")

    new_features = [
        "transaction_year",
        "transaction_month",
        "transaction_day",
        "transaction_hour",
        "credit_utilization",
        "amount_per_account_age",
        "foreign_high_risk",
        "velocity_amount_ratio",
    ]

    for feature in new_features:
        print(f"- {feature}")

    inspect_account_behavior(train_df)
    inspect_account_fraud_behavior(train_df)


if __name__ == "__main__":
    main()