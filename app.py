import streamlit as st
import pandas as pd
import os
import sys
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "models/investigation_cases_enriched.csv"
ACTIONS_PATH = "models/investigation_actions.csv"


# ============================================================
# GENAI IMPORT
# ============================================================

sys.path.append(
    os.path.join(
        os.path.dirname(__file__),
        "src"
    )
)

from genai_service import generate_investigation_summary


# ============================================================
# STREAMLIT CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="FraudLens",
    page_icon="🔎",
    layout="wide"
)


# ============================================================
# GENAI SESSION STATE
# ============================================================

if "genai_summary" not in st.session_state:
    st.session_state.genai_summary = None

if "genai_case_id" not in st.session_state:
    st.session_state.genai_case_id = None


# ============================================================
# LOAD MAIN INVESTIGATION DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["timestamp"]
    )

    return df


# ============================================================
# LOAD / CREATE INVESTIGATION ACTIONS
# ============================================================

def load_actions():

    if os.path.exists(ACTIONS_PATH):

        actions_df = pd.read_csv(
            ACTIONS_PATH
        )

    else:

        actions_df = pd.DataFrame(
            columns=[
                "case_id",
                "investigation_status",
                "investigator_decision",
                "investigation_notes",
                "updated_at"
            ]
        )

        actions_df.to_csv(
            ACTIONS_PATH,
            index=False
        )

    return actions_df


# ============================================================
# SAVE INVESTIGATION ACTION
# ============================================================

def save_action(
    case_id,
    investigation_status,
    investigator_decision,
    investigation_notes
):

    actions_df = load_actions()

    new_action = pd.DataFrame(
        [
            {
                "case_id": case_id,
                "investigation_status": investigation_status,
                "investigator_decision": investigator_decision,
                "investigation_notes": investigation_notes,
                "updated_at": datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            }
        ]
    )

    actions_df = actions_df[
        actions_df["case_id"].astype(str)
        != str(case_id)
    ]

    actions_df = pd.concat(
        [
            actions_df,
            new_action
        ],
        ignore_index=True
    )

    actions_df.to_csv(
        ACTIONS_PATH,
        index=False
    )

    return actions_df


# ============================================================
# LOAD DATA
# ============================================================

df = load_data()

actions_df = load_actions()


# ============================================================
# HEADER
# ============================================================

st.title("🔎 FraudLens")

st.subheader(
    "Real-Time Fraud Detection & Investigation Platform"
)

st.caption(
    "Model-driven fraud alerts with evidence and "
    "SHAP-based explanations."
)


# ============================================================
# MAIN MODEL KPIs
# ============================================================

total_cases = len(df)

fraud_cases = int(
    df["is_fraud"].sum()
)

false_alerts = (
    total_cases - fraud_cases
)

fraud_rate = (
    fraud_cases / total_cases
    if total_cases > 0
    else 0
)

critical_cases = int(
    (
        df["risk_level"]
        == "Critical"
    ).sum()
)


col1, col2, col3, col4, col5 = st.columns(5)


with col1:

    st.metric(
        "Investigation Cases",
        f"{total_cases:,}"
    )


with col2:

    st.metric(
        "Actual Fraud Cases",
        f"{fraud_cases:,}"
    )


with col3:

    st.metric(
        "False Alerts",
        f"{false_alerts:,}"
    )


with col4:

    st.metric(
        "Actual Fraud Rate",
        f"{fraud_rate:.2%}"
    )


with col5:

    st.metric(
        "Critical Cases",
        f"{critical_cases:,}"
    )


st.divider()


# ============================================================
# RISK OVERVIEW
# ============================================================

st.subheader("Risk Overview")

st.caption(
    "Distribution of model-generated investigation cases "
    "and their observed outcomes."
)


risk_order = [
    "Critical",
    "High",
    "Medium",
    "Low"
]


risk_distribution = (
    df["risk_level"]
    .value_counts()
    .reindex(
        risk_order,
        fill_value=0
    )
)


fraud_distribution = pd.Series(
    {
        "Fraud": fraud_cases,
        "Non-Fraud": false_alerts
    }
)


chart_col1, chart_col2 = st.columns(2)


with chart_col1:

    st.markdown(
        "#### Risk Level Distribution"
    )

    risk_chart_df = pd.DataFrame(
        {
            "Cases": risk_distribution
        }
    )

    st.bar_chart(
        risk_chart_df,
        horizontal=True,
        width="stretch"
    )


with chart_col2:

    st.markdown(
        "#### Actual Outcome Distribution"
    )

    outcome_chart_df = pd.DataFrame(
        {
            "Cases": fraud_distribution
        }
    )

    st.bar_chart(
        outcome_chart_df,
        width="stretch"
    )


# ============================================================
# FRAUD PROBABILITY DISTRIBUTION
# ============================================================

st.markdown(
    "#### Fraud Probability Distribution"
)


probability_bins = [
    0.90,
    0.92,
    0.94,
    0.96,
    0.98,
    1.00
]


probability_labels = [
    "0.90–0.92",
    "0.92–0.94",
    "0.94–0.96",
    "0.96–0.98",
    "0.98–1.00"
]


probability_categories = pd.cut(
    df["fraud_probability"],
    bins=probability_bins,
    labels=probability_labels,
    include_lowest=True
)


probability_distribution = (
    probability_categories
    .value_counts()
    .reindex(
        probability_labels,
        fill_value=0
    )
)


probability_chart_df = pd.DataFrame(
    {
        "Cases": probability_distribution
    }
)


st.bar_chart(
    probability_chart_df,
    width="stretch"
)


st.divider()


# ============================================================
# INVESTIGATION MANAGEMENT DASHBOARD
# ============================================================

st.subheader(
    "Investigation Management"
)

st.caption(
    "Operational view of investigator actions recorded "
    "for fraud investigation cases."
)


# ------------------------------------------------------------
# PREPARE ACTION DATA
# ------------------------------------------------------------

if len(actions_df) > 0:

    expected_action_columns = [
        "case_id",
        "investigation_status",
        "investigator_decision",
        "investigation_notes",
        "updated_at"
    ]

    for column in expected_action_columns:

        if column not in actions_df.columns:

            actions_df[column] = ""


    actions_df["case_id"] = (
        actions_df["case_id"]
        .astype(str)
    )


    management_df = actions_df.merge(
        df[
            [
                "case_id",
                "transaction_id",
                "account_id",
                "amount",
                "fraud_probability",
                "risk_level",
                "is_fraud"
            ]
        ],
        on="case_id",
        how="left"
    )

else:

    management_df = pd.DataFrame(
        columns=[
            "case_id",
            "investigation_status",
            "investigator_decision",
            "investigation_notes",
            "updated_at",
            "transaction_id",
            "account_id",
            "amount",
            "fraud_probability",
            "risk_level",
            "is_fraud"
        ]
    )


# ------------------------------------------------------------
# MANAGEMENT KPIs
# ------------------------------------------------------------

total_investigated = len(
    management_df
)


if total_investigated > 0:

    open_cases = int(
        (
            management_df[
                "investigation_status"
            ]
            == "Open"
        ).sum()
    )

    under_review_cases = int(
        (
            management_df[
                "investigation_status"
            ]
            == "Under Review"
        ).sum()
    )

    escalated_cases = int(
        (
            management_df[
                "investigation_status"
            ]
            == "Escalated"
        ).sum()
    )

    closed_cases = int(
        (
            management_df[
                "investigation_status"
            ]
            == "Closed"
        ).sum()
    )

    pending_decisions = int(
        (
            management_df[
                "investigator_decision"
            ]
            == "Pending"
        ).sum()
    )

    confirmed_fraud = int(
        (
            management_df[
                "investigator_decision"
            ]
            == "Confirm Fraud"
        ).sum()
    )

    marked_legitimate = int(
        (
            management_df[
                "investigator_decision"
            ]
            == "Mark as Legitimate"
        ).sum()
    )

    further_review = int(
        (
            management_df[
                "investigator_decision"
            ]
            == "Needs Further Review"
        ).sum()
    )

else:

    open_cases = 0
    under_review_cases = 0
    escalated_cases = 0
    closed_cases = 0
    pending_decisions = 0
    confirmed_fraud = 0
    marked_legitimate = 0
    further_review = 0


management_col1, management_col2, management_col3, management_col4 = st.columns(4)


with management_col1:

    st.metric(
        "Investigated",
        f"{total_investigated:,}"
    )


with management_col2:

    st.metric(
        "Open",
        f"{open_cases:,}"
    )


with management_col3:

    st.metric(
        "Under Review",
        f"{under_review_cases:,}"
    )


with management_col4:

    st.metric(
        "Closed",
        f"{closed_cases:,}"
    )


management_col5, management_col6, management_col7, management_col8 = st.columns(4)


with management_col5:

    st.metric(
        "Escalated",
        f"{escalated_cases:,}"
    )


with management_col6:

    st.metric(
        "Pending Decision",
        f"{pending_decisions:,}"
    )


with management_col7:

    st.metric(
        "Confirmed Fraud",
        f"{confirmed_fraud:,}"
    )


with management_col8:

    st.metric(
        "Further Review",
        f"{further_review:,}"
    )


# ------------------------------------------------------------
# MANAGEMENT CHARTS
# ------------------------------------------------------------

if total_investigated > 0:

    management_chart_col1, management_chart_col2 = st.columns(2)


    with management_chart_col1:

        st.markdown(
            "#### Investigation Status"
        )

        status_order = [
            "Open",
            "Under Review",
            "Escalated",
            "Closed"
        ]

        status_distribution = (
            management_df[
                "investigation_status"
            ]
            .value_counts()
            .reindex(
                status_order,
                fill_value=0
            )
        )

        status_chart_df = pd.DataFrame(
            {
                "Cases": status_distribution
            }
        )

        st.bar_chart(
            status_chart_df,
            width="stretch"
        )


    with management_chart_col2:

        st.markdown(
            "#### Investigator Decisions"
        )

        decision_order = [
            "Pending",
            "Confirm Fraud",
            "Mark as Legitimate",
            "Needs Further Review"
        ]

        decision_distribution = (
            management_df[
                "investigator_decision"
            ]
            .value_counts()
            .reindex(
                decision_order,
                fill_value=0
            )
        )

        decision_chart_df = pd.DataFrame(
            {
                "Cases": decision_distribution
            }
        )

        st.bar_chart(
            decision_chart_df,
            width="stretch"
        )


else:

    st.info(
        "No investigator actions have been recorded yet. "
        "Save an investigation from the Case Investigation "
        "section to populate this dashboard."
    )


st.divider()


# ============================================================
# MANAGEMENT FILTERS
# ============================================================

if total_investigated > 0:

    st.markdown(
        "#### Investigation Records"
    )


    management_filter_col1, management_filter_col2, management_filter_col3 = st.columns(3)


    with management_filter_col1:

        status_filter = st.multiselect(
            "Filter by Status",
            options=[
                "Open",
                "Under Review",
                "Escalated",
                "Closed"
            ],
            default=[
                "Open",
                "Under Review",
                "Escalated",
                "Closed"
            ]
        )


    with management_filter_col2:

        decision_filter = st.multiselect(
            "Filter by Decision",
            options=[
                "Pending",
                "Confirm Fraud",
                "Mark as Legitimate",
                "Needs Further Review"
            ],
            default=[
                "Pending",
                "Confirm Fraud",
                "Mark as Legitimate",
                "Needs Further Review"
            ]
        )


    with management_filter_col3:

        management_search = st.text_input(
            "Search Investigation",
            placeholder="Case / Transaction / Account..."
        )


    management_filtered = management_df[
        management_df[
            "investigation_status"
        ].isin(status_filter)
        &
        management_df[
            "investigator_decision"
        ].isin(decision_filter)
    ].copy()


    if management_search:

        management_search = (
            management_search
            .strip()
            .lower()
        )


        case_match = (
            management_filtered[
                "case_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                management_search,
                na=False
            )
        )


        transaction_match = (
            management_filtered[
                "transaction_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                management_search,
                na=False
            )
        )


        account_match = (
            management_filtered[
                "account_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                management_search,
                na=False
            )
        )


        management_filtered = management_filtered[
            case_match
            |
            transaction_match
            |
            account_match
        ]


    management_display_columns = [
        "case_id",
        "transaction_id",
        "account_id",
        "fraud_probability",
        "risk_level",
        "investigation_status",
        "investigator_decision",
        "updated_at"
    ]


    if len(management_filtered) > 0:

        management_display = (
            management_filtered[
                management_display_columns
            ]
            .sort_values(
                "updated_at",
                ascending=False
            )
            .copy()
        )


        management_display[
            "fraud_probability"
        ] = (
            management_display[
                "fraud_probability"
            ]
            .map(
                lambda x:
                f"{x:.2%}"
                if pd.notna(x)
                else "N/A"
            )
        )


        management_display.columns = [
            "Case",
            "Transaction",
            "Account",
            "Fraud Probability",
            "Risk Level",
            "Status",
            "Decision",
            "Last Updated"
        ]


        st.dataframe(
            management_display,
            width="stretch",
            hide_index=True,
            height=350
        )


    else:

        st.warning(
            "No investigation records match "
            "the selected filters."
        )


st.divider()


# ============================================================
# SIDEBAR FILTERS FOR FRAUD QUEUE
# ============================================================

st.sidebar.header(
    "Investigation Filters"
)


risk_options = sorted(
    df["risk_level"]
    .dropna()
    .unique()
)


selected_risk = st.sidebar.multiselect(
    "Risk Level",
    options=risk_options,
    default=risk_options
)


min_probability = st.sidebar.slider(
    "Minimum Fraud Probability",
    min_value=0.90,
    max_value=1.00,
    value=0.90,
    step=0.01
)


search_value = st.sidebar.text_input(
    "Search Case / Transaction",
    placeholder="FL-000001 or TXN..."
)


# ============================================================
# APPLY FRAUD QUEUE FILTERS
# ============================================================

filtered_df = df[
    df["risk_level"].isin(
        selected_risk
    )
    &
    (
        df["fraud_probability"]
        >= min_probability
    )
].copy()


if search_value:

    search_value = (
        search_value
        .strip()
        .lower()
    )


    case_match = (
        filtered_df[
            "case_id"
        ]
        .astype(str)
        .str.lower()
        .str.contains(
            search_value,
            na=False
        )
    )


    transaction_match = (
        filtered_df[
            "transaction_id"
        ]
        .astype(str)
        .str.lower()
        .str.contains(
            search_value,
            na=False
        )
    )


    filtered_df = filtered_df[
        case_match
        |
        transaction_match
    ]


# ============================================================
# INVESTIGATION QUEUE
# ============================================================

st.subheader(
    "Investigation Queue"
)

st.caption(
    "Transactions currently flagged by the fraud detection model."
)


queue_col1, queue_col2, queue_col3, queue_col4 = st.columns(4)


filtered_count = len(
    filtered_df
)


filtered_fraud = (
    int(
        filtered_df["is_fraud"].sum()
    )
    if filtered_count > 0
    else 0
)


filtered_rate = (
    filtered_fraud
    / filtered_count
    if filtered_count > 0
    else 0
)


average_probability = (
    filtered_df[
        "fraud_probability"
    ].mean()
    if filtered_count > 0
    else 0
)


with queue_col1:

    st.metric(
        "Filtered Cases",
        f"{filtered_count:,}"
    )


with queue_col2:

    st.metric(
        "Actual Fraud",
        f"{filtered_fraud:,}"
    )


with queue_col3:

    st.metric(
        "Fraud Rate",
        f"{filtered_rate:.2%}"
    )


with queue_col4:

    st.metric(
        "Average Probability",
        f"{average_probability:.2%}"
    )


# ============================================================
# QUEUE TABLE
# ============================================================

display_columns = [
    "case_id",
    "transaction_id",
    "account_id",
    "timestamp",
    "amount",
    "fraud_probability",
    "risk_level",
    "investigation_priority",
    "evidence_count",
    "is_fraud"
]


if filtered_count > 0:

    queue_display = (
        filtered_df[
            display_columns
        ]
        .sort_values(
            "fraud_probability",
            ascending=False
        )
        .copy()
    )


    queue_display["amount"] = (
        queue_display["amount"]
        .map(
            lambda x:
            f"${x:,.2f}"
        )
    )


    queue_display[
        "fraud_probability"
    ] = (
        queue_display[
            "fraud_probability"
        ]
        .map(
            lambda x:
            f"{x:.2%}"
        )
    )


    queue_display["is_fraud"] = (
        queue_display["is_fraud"]
        .map(
            lambda x:
            "Yes" if x == 1 else "No"
        )
    )


    queue_display.columns = [
        "Case",
        "Transaction",
        "Account",
        "Timestamp",
        "Amount",
        "Fraud Probability",
        "Risk Level",
        "Priority",
        "Evidence",
        "Actual Fraud"
    ]


    st.dataframe(
        queue_display,
        width="stretch",
        hide_index=True,
        height=420
    )


else:

    st.warning(
        "No investigation cases match "
        "the selected filters."
    )


st.divider()


# ============================================================
# CASE INVESTIGATION
# ============================================================

st.subheader(
    "Case Investigation"
)

st.caption(
    "Inspect transaction details, investigation evidence, "
    "and SHAP-based model explanations."
)


if filtered_count > 0:

    # --------------------------------------------------------
    # CASE SELECTION
    # --------------------------------------------------------

    selected_case_id = st.selectbox(
        "Select a case",
        options=filtered_df[
            "case_id"
        ].tolist()
    )


    case = (
        filtered_df[
            filtered_df["case_id"]
            == selected_case_id
        ]
        .iloc[0]
    )


    case_id = str(
        case["case_id"]
    )


    # ========================================================
    # RESET AI SUMMARY WHEN CASE CHANGES
    # ========================================================

    if st.session_state.genai_case_id != case_id:

        st.session_state.genai_case_id = case_id

        st.session_state.genai_summary = None


    st.markdown(
        f"### Case {case_id}"
    )


    # --------------------------------------------------------
    # CASE KPIs
    # --------------------------------------------------------

    case_col1, case_col2, case_col3, case_col4 = st.columns(4)


    with case_col1:

        st.metric(
            "Fraud Probability",
            f"{case['fraud_probability']:.2%}"
        )


    with case_col2:

        st.metric(
            "Risk Level",
            case["risk_level"]
        )


    with case_col3:

        st.metric(
            "Priority",
            case["investigation_priority"]
        )


    with case_col4:

        actual_status = (
            "Fraud"
            if case["is_fraud"] == 1
            else "Non-Fraud"
        )


        st.metric(
            "Actual Outcome",
            actual_status
        )


    # --------------------------------------------------------
    # TRANSACTION DETAILS
    # --------------------------------------------------------

    st.markdown(
        "### Transaction Details"
    )


    detail_col1, detail_col2 = st.columns(2)


    with detail_col1:

        st.write(
            f"**Transaction ID:** "
            f"{case['transaction_id']}"
        )

        st.write(
            f"**Account ID:** "
            f"{case['account_id']}"
        )

        st.write(
            f"**Timestamp:** "
            f"{case['timestamp']}"
        )

        st.write(
            f"**Amount:** "
            f"${case['amount']:,.2f}"
        )

        st.write(
            f"**Merchant Category:** "
            f"{case['merchant_category']}"
        )

        st.write(
            f"**Merchant Country:** "
            f"{case['merchant_country']}"
        )


    with detail_col2:

        st.write(
            f"**Device Type:** "
            f"{case['device_type']}"
        )

        st.write(
            f"**Device Known:** "
            f"{'Yes' if case['device_known'] == 1 else 'No'}"
        )

        st.write(
            f"**IP Risk Score:** "
            f"{case['ip_risk_score']}"
        )

        st.write(
            f"**Foreign Transaction:** "
            f"{'Yes' if case['is_foreign_txn'] == 1 else 'No'}"
        )

        st.write(
            f"**2FA Used:** "
            f"{'Yes' if case['has_2fa'] == 1 else 'No'}"
        )

        st.write(
            f"**Card Present:** "
            f"{'Yes' if case['card_present'] == 1 else 'No'}"
        )


    # --------------------------------------------------------
    # BEHAVIOURAL SIGNALS
    # --------------------------------------------------------

    st.markdown(
        "### Behavioural Signals"
    )


    behavior_col1, behavior_col2, behavior_col3, behavior_col4 = st.columns(4)


    with behavior_col1:

        st.metric(
            "Transaction Velocity",
            f"{case['velocity_1h']:.2f}"
        )


    with behavior_col2:

        st.metric(
            "Amount / Average",
            f"{case['amount_vs_avg_ratio']:.2f}x"
        )


    with behavior_col3:

        st.metric(
            "Time Since Last Txn",
            f"{case['time_since_last_s']:.0f}s"
        )


    with behavior_col4:

        st.metric(
            "Account Age",
            f"{case['account_age_days']:.0f} days"
        )


    # --------------------------------------------------------
    # INVESTIGATION EVIDENCE
    # --------------------------------------------------------

    st.markdown(
        "### Investigation Evidence"
    )


    evidence = str(
        case["investigation_evidence"]
    )


    evidence_items = [
        item.strip()
        for item in evidence.split("|")
        if item.strip()
    ]


    if evidence_items:

        for item in evidence_items:

            st.markdown(
                f"- ⚠️ {item}"
            )

    else:

        st.info(
            "No rule-based evidence signals were detected."
        )


    # --------------------------------------------------------
    # SHAP MODEL EXPLANATION
    # --------------------------------------------------------

    st.markdown(
        "### Model Explanation"
    )


    st.caption(
        "SHAP shows which model features contributed "
        "most strongly to this prediction."
    )


    shap_rows = []


    for rank in range(1, 6):

        feature = case[
            f"top_feature_{rank}"
        ]

        shap_value = case[
            f"top_feature_{rank}_shap"
        ]

        impact = case[
            f"top_feature_{rank}_impact"
        ]


        shap_rows.append(
            {
                "Rank": rank,
                "Feature": feature,
                "SHAP Value": shap_value,
                "Impact": impact
            }
        )


    shap_display = pd.DataFrame(
        shap_rows
    )


    shap_col1, shap_col2 = st.columns(2)


    with shap_col1:

        st.dataframe(
            shap_display,
            width="stretch",
            hide_index=True
        )


    with shap_col2:

        shap_chart = (
            shap_display[
                [
                    "Feature",
                    "SHAP Value"
                ]
            ]
            .set_index("Feature")
        )


        st.bar_chart(
            shap_chart,
            horizontal=True,
            width="stretch"
        )


    # ========================================================
    # GENAI INVESTIGATION ASSISTANT
    # ========================================================

    st.markdown(
        "### 🤖 AI Investigation Assistant"
    )

    st.caption(
        "Uses the existing model prediction, investigation "
        "evidence, and SHAP explanations to generate a "
        "human-readable investigation summary."
    )

    st.info(
        "The AI assistant does not make the fraud decision. "
        "The XGBoost model remains responsible for the "
        "fraud prediction."
    )


    # --------------------------------------------------------
    # GENERATE AI SUMMARY BUTTON
    # --------------------------------------------------------

    if st.button(
        "Generate AI Investigation Summary",
        type="primary",
        key=f"genai_{case_id}"
    ):

        # Clear previous result before generating
        st.session_state.genai_summary = None

        with st.spinner(
            "Generating investigation summary..."
        ):

            try:

                result = generate_investigation_summary(
                    case
                )


                if result["success"]:

                    # Save AI response in session state
                    st.session_state.genai_summary = (
                        result["summary"]
                    )


                else:

                    error_type = result.get(
                        "error_type",
                        "ERROR"
                    )

                    message = result.get(
                        "message",
                        "Unable to generate AI summary."
                    )


                    if error_type == "RATE_LIMIT":

                        st.warning(
                            "Gemini is temporarily "
                            "unavailable because the API "
                            "rate limit has been reached."
                        )


                    elif error_type == "NO_API_KEY":

                        st.error(
                            "Gemini API key was not found. "
                            "Check your .env file."
                        )


                    else:

                        st.error(
                            f"AI assistant error: {message}"
                        )


            except Exception as e:

                st.error(
                    "The AI assistant could not generate "
                    "a summary. The rest of FraudLens "
                    "continues to work normally."
                )

                st.caption(
                    f"Technical details: {str(e)}"
                )


    # --------------------------------------------------------
    # DISPLAY SAVED AI SUMMARY
    # --------------------------------------------------------

    if (
        st.session_state.genai_case_id == case_id
        and st.session_state.genai_summary
    ):

        st.markdown(
            "#### 📋 Generated Investigation Summary"
        )

        st.markdown(
            st.session_state.genai_summary
        )


    # ========================================================
    # INVESTIGATOR ACTION
    # ========================================================

    st.markdown(
        "### Investigator Action"
    )


    existing_action = actions_df[
        actions_df["case_id"].astype(str)
        == case_id
    ]


    if len(existing_action) > 0:

        previous_action = (
            existing_action
            .iloc[-1]
        )


        default_status = (
            previous_action[
                "investigation_status"
            ]
        )


        default_decision = (
            previous_action[
                "investigator_decision"
            ]
        )


        default_notes = (
            previous_action[
                "investigation_notes"
            ]
        )

    else:

        default_status = "Open"

        default_decision = "Pending"

        default_notes = ""


    status_options = [
        "Open",
        "Under Review",
        "Escalated",
        "Closed"
    ]


    decision_options = [
        "Pending",
        "Confirm Fraud",
        "Mark as Legitimate",
        "Needs Further Review"
    ]


    action_col1, action_col2 = st.columns(2)


    with action_col1:

        investigation_status = st.selectbox(
            "Investigation Status",
            status_options,
            index=(
                status_options.index(
                    default_status
                )
                if default_status
                in status_options
                else 0
            ),
            key=f"status_{case_id}"
        )


        investigator_decision = st.selectbox(
            "Investigator Decision",
            decision_options,
            index=(
                decision_options.index(
                    default_decision
                )
                if default_decision
                in decision_options
                else 0
            ),
            key=f"decision_{case_id}"
        )


    with action_col2:

        investigation_notes = st.text_area(
            "Investigation Notes",
            value=(
                default_notes
                if pd.notna(default_notes)
                else ""
            ),
            placeholder=(
                "Enter investigation findings "
                "or comments..."
            ),
            height=120,
            key=f"notes_{case_id}"
        )


    # --------------------------------------------------------
    # SAVE INVESTIGATION
    # --------------------------------------------------------

    if st.button(
        "Save Investigation",
        type="primary",
        key=f"save_{case_id}"
    ):

        save_action(
            case_id=case_id,
            investigation_status=(
                investigation_status
            ),
            investigator_decision=(
                investigator_decision
            ),
            investigation_notes=(
                investigation_notes
            )
        )


        st.success(
            f"Investigation saved successfully "
            f"for {case_id}."
        )


        st.rerun()


    # ========================================================
    # COMPLETE CASE DATA
    # ========================================================

    with st.expander(
        "View complete case data"
    ):

        raw_case = (
            case
            .to_frame(name="Value")
            .copy()
        )


        raw_case["Value"] = (
            raw_case["Value"]
            .astype(str)
        )


        st.dataframe(
            raw_case,
            width="stretch",
            hide_index=False
        )


else:

    st.info(
        "Select filters that return at least one case."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "FraudLens | ML Risk Scoring • "
    "Rule-Based Evidence • "
    "SHAP Explainability • "
    "GenAI Investigation Assistant • "
    "Investigation Workflow"
)