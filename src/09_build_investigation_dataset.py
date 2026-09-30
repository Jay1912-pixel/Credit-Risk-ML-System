import pandas as pd


# ============================================================
# PATHS
# ============================================================

INVESTIGATION_PATH = (
    "models/investigation_cases.csv"
)

SHAP_PATH = (
    "models/investigation_case_explanations.csv"
)

OUTPUT_PATH = (
    "models/investigation_cases_enriched.csv"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("FRAUDLENS - BUILD INVESTIGATION DATASET")
    print("=" * 60)

    # --------------------------------------------------------
    # Load investigation cases
    # --------------------------------------------------------

    print("\nLoading investigation cases...")

    investigation_df = pd.read_csv(
        INVESTIGATION_PATH
    )

    print(
        f"Investigation data shape: "
        f"{investigation_df.shape}"
    )

    # --------------------------------------------------------
    # Load SHAP explanations
    # --------------------------------------------------------

    print("\nLoading SHAP explanations...")

    shap_df = pd.read_csv(
        SHAP_PATH
    )

    print(
        f"SHAP data shape: "
        f"{shap_df.shape}"
    )

    # --------------------------------------------------------
    # Validate required keys
    # --------------------------------------------------------

    required_keys = [
        "case_id",
        "transaction_id"
    ]

    for key in required_keys:

        if key not in investigation_df.columns:

            raise ValueError(
                f"Missing '{key}' "
                "from investigation data."
            )

        if key not in shap_df.columns:

            raise ValueError(
                f"Missing '{key}' "
                "from SHAP data."
            )

    # --------------------------------------------------------
    # Check uniqueness
    # --------------------------------------------------------

    if investigation_df[
        "case_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate case IDs found "
            "in investigation data."
        )

    if shap_df[
        "case_id"
    ].duplicated().any():

        raise ValueError(
            "Duplicate case IDs found "
            "in SHAP data."
        )

    # --------------------------------------------------------
    # Check case counts
    # --------------------------------------------------------

    if len(investigation_df) != len(shap_df):

        raise ValueError(
            "Investigation cases and SHAP "
            "case counts do not match."
        )

    print(
        "\nCase count validation: PASS"
    )

    # --------------------------------------------------------
    # Validate case ID matching
    # --------------------------------------------------------

    investigation_ids = set(
        investigation_df["case_id"]
    )

    shap_ids = set(
        shap_df["case_id"]
    )

    if investigation_ids != shap_ids:

        missing_in_shap = (
            investigation_ids - shap_ids
        )

        missing_in_investigation = (
            shap_ids - investigation_ids
        )

        raise ValueError(
            "Case IDs do not match.\n"
            f"Missing in SHAP: "
            f"{len(missing_in_shap)}\n"
            f"Missing in investigation: "
            f"{len(missing_in_investigation)}"
        )

    print(
        "Case ID matching: PASS"
    )

    # --------------------------------------------------------
    # Validate transaction IDs
    # --------------------------------------------------------

    merged_check = investigation_df[
        [
            "case_id",
            "transaction_id"
        ]
    ].merge(
        shap_df[
            [
                "case_id",
                "transaction_id"
            ]
        ],
        on="case_id",
        how="left",
        suffixes=(
            "_investigation",
            "_shap"
        )
    )

    transaction_match = (
        merged_check[
            "transaction_id_investigation"
        ]
        ==
        merged_check[
            "transaction_id_shap"
        ]
    )

    if not transaction_match.all():

        mismatch_count = (
            (~transaction_match).sum()
        )

        raise ValueError(
            f"{mismatch_count} transaction IDs "
            "do not match between datasets."
        )

    print(
        "Transaction ID matching: PASS"
    )

    # --------------------------------------------------------
    # Select SHAP columns
    # --------------------------------------------------------

    shap_columns = [
        "case_id",
        "top_feature_1",
        "top_feature_1_shap",
        "top_feature_1_impact",
        "top_feature_2",
        "top_feature_2_shap",
        "top_feature_2_impact",
        "top_feature_3",
        "top_feature_3_shap",
        "top_feature_3_impact",
        "top_feature_4",
        "top_feature_4_shap",
        "top_feature_4_impact",
        "top_feature_5",
        "top_feature_5_shap",
        "top_feature_5_impact"
    ]

    shap_selected = shap_df[
        shap_columns
    ].copy()

    # --------------------------------------------------------
    # Merge
    # --------------------------------------------------------

    enriched_df = investigation_df.merge(
        shap_selected,
        on="case_id",
        how="left",
        validate="one_to_one"
    )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    if len(enriched_df) != len(
        investigation_df
    ):

        raise ValueError(
            "Enriched dataset row count "
            "changed after merge."
        )

    if enriched_df[
        "transaction_id"
    ].isnull().any():

        raise ValueError(
            "Null transaction IDs found "
            "after enrichment."
        )

    if enriched_df[
        "top_feature_1"
    ].isnull().any():

        raise ValueError(
            "Missing SHAP explanations "
            "after enrichment."
        )

    print(
        "\nEnrichment validation: PASS"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    enriched_df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("INVESTIGATION DATASET COMPLETE")
    print("=" * 60)

    print(
        f"\nFinal shape: "
        f"{enriched_df.shape}"
    )

    print(
        f"\nTotal investigation cases: "
        f"{len(enriched_df):,}"
    )

    print(
        "\nFinal columns:"
    )

    for column in enriched_df.columns:
        print(
            f"- {column}"
        )

    print(
        "\nSaved:"
    )

    print(
        f"- {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()