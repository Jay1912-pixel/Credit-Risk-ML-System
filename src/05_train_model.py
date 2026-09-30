import os
import json
import pandas as pd

from xgboost import XGBClassifier

from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


TRAIN_PATH = "data/processed/train_model.csv"
TEST_PATH = "data/processed/test_model.csv"

MODEL_DIR = "models"
MODEL_PATH = "models/fraud_xgb.json"
THRESHOLD_PATH = "models/threshold.json"


def evaluate_threshold(y_true, probabilities, threshold):
    """
    Evaluate model performance at a specific fraud probability threshold.
    """

    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions
    ).ravel()

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp
    }


def prepare_categories(X_train, X_validation, X_test):
    """
    Make categorical columns use the same category definitions
    across train, validation and test.
    """

    categorical_columns = X_train.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    print("\nCategorical Columns:")

    for column in categorical_columns:
        print(f"- {column}")

    for column in categorical_columns:

        categories = pd.concat(
            [
                X_train[column],
                X_validation[column],
                X_test[column]
            ]
        ).astype("category").cat.categories

        X_train[column] = pd.Categorical(
            X_train[column],
            categories=categories
        )

        X_validation[column] = pd.Categorical(
            X_validation[column],
            categories=categories
        )

        X_test[column] = pd.Categorical(
            X_test[column],
            categories=categories
        )

    return (
        X_train,
        X_validation,
        X_test,
        categorical_columns
    )


def main():

    print("=" * 60)
    print("FRAUDLENS - MODEL TRAINING & EVALUATION")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. CREATE MODEL DIRECTORY
    # ---------------------------------------------------------

    os.makedirs(MODEL_DIR, exist_ok=True)

    # ---------------------------------------------------------
    # 2. LOAD DATA
    # ---------------------------------------------------------

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    print(f"\nOriginal Training Shape: {train_df.shape}")
    print(f"Final Test Shape: {test_df.shape}")

    # ---------------------------------------------------------
    # 3. CHRONOLOGICAL TRAIN / VALIDATION SPLIT
    # ---------------------------------------------------------

    train_size = int(len(train_df) * 0.80)

    train_part = train_df.iloc[:train_size].copy()
    validation_part = train_df.iloc[train_size:].copy()

    print("\n" + "=" * 60)
    print("CHRONOLOGICAL DATA SPLIT")
    print("=" * 60)

    print(f"\nTraining Data:   {train_part.shape}")
    print(f"Validation Data: {validation_part.shape}")
    print(f"Final Test Data: {test_df.shape}")

    # ---------------------------------------------------------
    # 4. TARGET SEPARATION
    # ---------------------------------------------------------

    target = "is_fraud"

    X_train = train_part.drop(columns=[target])
    y_train = train_part[target]

    X_validation = validation_part.drop(columns=[target])
    y_validation = validation_part[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # ---------------------------------------------------------
    # 5. HANDLE CATEGORICAL FEATURES
    # ---------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        categorical_columns
    ) = prepare_categories(
        X_train,
        X_validation,
        X_test
    )

    # ---------------------------------------------------------
    # 6. CLASS IMBALANCE
    # ---------------------------------------------------------

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    scale_pos_weight = negative_count / positive_count

    print("\n" + "=" * 60)
    print("CLASS IMBALANCE")
    print("=" * 60)

    print(
        f"\nTraining Non-Fraud Transactions: "
        f"{negative_count:,}"
    )

    print(
        f"Training Fraud Transactions: "
        f"{positive_count:,}"
    )

    print(
        f"Scale Pos Weight: "
        f"{scale_pos_weight:.2f}"
    )

    # ---------------------------------------------------------
    # 7. CREATE XGBOOST MODEL
    # ---------------------------------------------------------

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
        tree_method="hist",
        random_state=42,
        n_jobs=-1
    )

    # ---------------------------------------------------------
    # 8. TRAIN MODEL
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING XGBOOST")
    print("=" * 60)

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (X_validation, y_validation)
        ],
        verbose=False
    )

    print("\nTraining completed successfully.")

    # ---------------------------------------------------------
    # 9. VALIDATION PREDICTIONS
    # ---------------------------------------------------------

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    validation_pr_auc = average_precision_score(
        y_validation,
        validation_probabilities
    )

    print("\n" + "=" * 60)
    print("VALIDATION PERFORMANCE")
    print("=" * 60)

    print(
        f"\nValidation PR-AUC: "
        f"{validation_pr_auc:.4f}"
    )

    # ---------------------------------------------------------
    # 10. THRESHOLD ANALYSIS
    # ---------------------------------------------------------

    thresholds = [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90
    ]

    validation_results = []

    for threshold in thresholds:

        result = evaluate_threshold(
            y_validation,
            validation_probabilities,
            threshold
        )

        validation_results.append(result)

    validation_results_df = pd.DataFrame(
        validation_results
    )

    print("\n" + "=" * 60)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 60)

    print(
        "\n" +
        validation_results_df.to_string(
            index=False,
            formatters={
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format
            }
        )
    )

    # ---------------------------------------------------------
    # 11. SELECT THRESHOLD
    # ---------------------------------------------------------

    best_validation_row = validation_results_df.loc[
        validation_results_df["f1"].idxmax()
    ]

    selected_threshold = float(
        best_validation_row["threshold"]
    )

    print("\n" + "=" * 60)
    print("SELECTED THRESHOLD FROM VALIDATION")
    print("=" * 60)

    print(
        f"\nSelected Threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Validation Precision: "
        f"{best_validation_row['precision']:.4f}"
    )

    print(
        f"Validation Recall: "
        f"{best_validation_row['recall']:.4f}"
    )

    print(
        f"Validation F1: "
        f"{best_validation_row['f1']:.4f}"
    )

    # ---------------------------------------------------------
    # 12. FINAL TEST EVALUATION
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION")
    print("=" * 60)

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    test_pr_auc = average_precision_score(
        y_test,
        test_probabilities
    )

    final_test_result = evaluate_threshold(
        y_test,
        test_probabilities,
        selected_threshold
    )

    print(
        f"\nFinal Test PR-AUC: "
        f"{test_pr_auc:.4f}"
    )

    print(
        f"Final Test Threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        f"Final Test Precision: "
        f"{final_test_result['precision']:.4f}"
    )

    print(
        f"Final Test Recall: "
        f"{final_test_result['recall']:.4f}"
    )

    print(
        f"Final Test F1: "
        f"{final_test_result['f1']:.4f}"
    )

    print(
        f"Final Test False Positives: "
        f"{final_test_result['false_positives']:,}"
    )

    print(
        f"Final Test False Negatives: "
        f"{final_test_result['false_negatives']:,}"
    )

    print(
        f"Final Test True Positives: "
        f"{final_test_result['true_positives']:,}"
    )

    # ---------------------------------------------------------
    # 13. SAVE MODEL
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    model.save_model(MODEL_PATH)

    print(
        f"\nModel saved successfully:"
    )

    print(f"- {MODEL_PATH}")

    # ---------------------------------------------------------
    # 14. SAVE THRESHOLD CONFIGURATION
    # ---------------------------------------------------------

    threshold_config = {
        "threshold": selected_threshold,
        "selection_metric": "F1",
        "validation_f1": float(
            best_validation_row["f1"]
        ),
        "validation_precision": float(
            best_validation_row["precision"]
        ),
        "validation_recall": float(
            best_validation_row["recall"]
        )
    }

    with open(
        THRESHOLD_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            threshold_config,
            file,
            indent=4
        )

    print(
        f"Threshold configuration saved:"
    )

    print(f"- {THRESHOLD_PATH}")

    # ---------------------------------------------------------
    # 15. FINAL SUMMARY
    # ---------------------------------------------------------

    print("\n" + "=" * 60)
    print("FRAUDLENS MODEL SUMMARY")
    print("=" * 60)

    print("\nData Strategy:")
    print("- Chronological training data")
    print("- 80% training")
    print("- 20% validation")
    print("- Future test period kept separate")

    print("\nThreshold Strategy:")
    print("- Threshold selected using validation data")
    print("- Final performance evaluated on untouched test data")

    print("\nSelected Threshold:")
    print(f"- {selected_threshold:.2f}")

    print("\nFinal Test Metrics:")
    print(
        f"- PR-AUC: "
        f"{test_pr_auc:.4f}"
    )

    print(
        f"- Precision: "
        f"{final_test_result['precision']:.4f}"
    )

    print(
        f"- Recall: "
        f"{final_test_result['recall']:.4f}"
    )

    print(
        f"- F1: "
        f"{final_test_result['f1']:.4f}"
    )

    print("\nSaved Artifacts:")
    print(f"- {MODEL_PATH}")
    print(f"- {THRESHOLD_PATH}")


if __name__ == "__main__":
    main()