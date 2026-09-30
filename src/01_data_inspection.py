import pandas as pd

DATA_PATH = "data/raw/transactions.csv"
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df


def time_based_split(df):
    df = df.sort_values("timestamp").reset_index(drop=True)

    split_index = int(len(df) * 0.80)

    train_df = df.iloc[:split_index].copy()
    test_df = df.iloc[split_index:].copy()

    return train_df, test_df


def main():
    df = load_data()

    train_df, test_df = time_based_split(df)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print("=" * 60)
    print("FRAUDLENS - DATA SPLIT")
    print("=" * 60)

    print(f"\nFull Dataset: {df.shape}")
    print(f"Training Set: {train_df.shape}")
    print(f"Test Set: {test_df.shape}")

    print("\nTraining Period:")
    print(train_df["timestamp"].min())
    print(train_df["timestamp"].max())

    print("\nTest Period:")
    print(test_df["timestamp"].min())
    print(test_df["timestamp"].max())

    print("\nTraining Fraud Rate:")
    print(f"{train_df['is_fraud'].mean() * 100:.2f}%")

    print("\nTest Fraud Rate:")
    print(f"{test_df['is_fraud'].mean() * 100:.2f}%")

    print("\nSaved:")
    print(TRAIN_PATH)
    print(TEST_PATH)


if __name__ == "__main__":
    main()