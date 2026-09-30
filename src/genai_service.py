import os
import pandas as pd
from dotenv import load_dotenv
from google import genai


# ============================================================
# LOAD ENVIRONMENT
# ============================================================

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

MODEL_NAME = "gemini-3.5-flash"


# ============================================================
# GEMINI CLIENT
# ============================================================

if API_KEY:
    client = genai.Client(
        api_key=API_KEY
    )
else:
    client = None


# ============================================================
# BUILD CASE CONTEXT
# ============================================================

def build_case_context(case):

    evidence = str(
        case["investigation_evidence"]
    )

    evidence_items = [
        item.strip()
        for item in evidence.split("|")
        if item.strip()
    ]


    shap_features = []


    for rank in range(1, 6):

        feature = case.get(
            f"top_feature_{rank}"
        )

        shap_value = case.get(
            f"top_feature_{rank}_shap"
        )

        impact = case.get(
            f"top_feature_{rank}_impact"
        )


        if pd.notna(feature):

            shap_features.append(
                {
                    "feature": str(feature),
                    "shap_value": round(
                        float(shap_value),
                        4
                    ),
                    "impact": str(impact)
                }
            )


    context = {

        "case_id": str(
            case["case_id"]
        ),

        "transaction_id": str(
            case["transaction_id"]
        ),

        "account_id": str(
            case["account_id"]
        ),

        "timestamp": str(
            case["timestamp"]
        ),

        "amount": round(
            float(case["amount"]),
            2
        ),

        "fraud_probability": round(
            float(case["fraud_probability"]),
            4
        ),

        "risk_level": str(
            case["risk_level"]
        ),

        "investigation_priority": str(
            case["investigation_priority"]
        ),

        "merchant_category": str(
            case["merchant_category"]
        ),

        "merchant_country": str(
            case["merchant_country"]
        ),

        "device_type": str(
            case["device_type"]
        ),

        "device_known": int(
            case["device_known"]
        ),

        "ip_risk_score": float(
            case["ip_risk_score"]
        ),

        "is_foreign_txn": int(
            case["is_foreign_txn"]
        ),

        "velocity_1h": float(
            case["velocity_1h"]
        ),

        "amount_vs_avg_ratio": float(
            case["amount_vs_avg_ratio"]
        ),

        "time_since_last_s": float(
            case["time_since_last_s"]
        ),

        "account_age_days": float(
            case["account_age_days"]
        ),

        "has_2fa": int(
            case["has_2fa"]
        ),

        "card_present": int(
            case["card_present"]
        ),

        "rule_based_evidence": evidence_items,

        "shap_explanations": shap_features
    }


    return context


# ============================================================
# GENERATE INVESTIGATION SUMMARY
# ============================================================

def generate_investigation_summary(case):

    # --------------------------------------------------------
    # CHECK API KEY
    # --------------------------------------------------------

    if client is None:

        return {
            "success": False,
            "error_type": "NO_API_KEY",
            "message": (
                "GEMINI_API_KEY was not found. "
                "Check the .env file."
            )
        }


    # --------------------------------------------------------
    # BUILD CONTEXT
    # --------------------------------------------------------

    context = build_case_context(
        case
    )


    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an investigation assistant inside
a fraud detection platform called FraudLens.

Your job is to help a human investigator
understand an already-flagged transaction.

IMPORTANT RULES:

1. The XGBoost model is responsible for
   the fraud prediction.

2. Do NOT change or recalculate the
   fraud probability.

3. Do NOT independently declare the
   transaction fraudulent.

4. Do NOT invent any evidence.

5. Use ONLY the information provided
   in the case data below.

6. Clearly distinguish:
   - XGBoost model output
   - Rule-based investigation evidence
   - SHAP model explanation

7. Explain technical information
   in simple language.

8. Provide useful investigation
   focus areas for a human investigator.

9. Mention limitations where appropriate.

Return the response using exactly
these sections:

## INVESTIGATION SUMMARY

Give a short summary of the case.

## KEY EVIDENCE

Explain the important rule-based
signals found in the transaction.

## MODEL EXPLANATION

Explain the fraud probability and
the main SHAP contributors.

## INVESTIGATION FOCUS

Give practical things an investigator
should verify.

## LIMITATIONS

Mention what cannot be concluded
from the available information.

CASE DATA:

{context}
"""


    # --------------------------------------------------------
    # GEMINI API CALL
    # --------------------------------------------------------

    try:

        response = client.models.generate_content(

            model=MODEL_NAME,

            contents=prompt
        )


        # ----------------------------------------------------
        # GET RESPONSE TEXT
        # ----------------------------------------------------

        output = getattr(
            response,
            "text",
            None
        )


        if not output:

            return {
                "success": False,
                "error_type": "EMPTY_RESPONSE",
                "message": (
                    "Gemini returned an empty response."
                )
            }


        return {
            "success": True,
            "summary": output
        }


    # --------------------------------------------------------
    # ERROR HANDLING
    # --------------------------------------------------------

    except Exception as e:

        error_text = str(e)

        print(
            "\n========== GEMINI ERROR =========="
        )

        print(error_text)

        print(
            "==================================\n"
        )


        # ----------------------------------------------------
        # RATE LIMIT
        # ----------------------------------------------------

        if (
            "429" in error_text
            or "rate limit" in error_text.lower()
            or "too_many_requests"
            in error_text.lower()
        ):

            return {
                "success": False,
                "error_type": "RATE_LIMIT",
                "message": (
                    "Gemini API rate limit has "
                    "been reached. The dashboard "
                    "is working, but Gemini is "
                    "currently unavailable."
                )
            }


        # ----------------------------------------------------
        # MODEL ERROR
        # ----------------------------------------------------

        if (
            "404" in error_text
            or "NOT_FOUND"
            in error_text
        ):

            return {
                "success": False,
                "error_type": "MODEL_ERROR",
                "message": (
                    f"The Gemini model "
                    f"'{MODEL_NAME}' is unavailable."
                )
            }


        # ----------------------------------------------------
        # API KEY ERROR
        # ----------------------------------------------------

        if (
            "401" in error_text
            or "403" in error_text
            or "API key" in error_text
        ):

            return {
                "success": False,
                "error_type": "AUTH_ERROR",
                "message": (
                    "Gemini API authentication "
                    "failed. Check GEMINI_API_KEY."
                )
            }


        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        return {
            "success": False,
            "error_type": "API_ERROR",
            "message": error_text
        }