# FraudLens — Real-Time Fraud Detection & Investigation Platform

FraudLens is an end-to-end fraud detection and investigation platform built using machine learning, explainability, rule-based investigation evidence, and an investigator-facing Streamlit dashboard.

The system is designed to answer three practical questions:

1. Is this transaction likely to be fraudulent?
2. Why does the model consider it suspicious?
3. What evidence should an investigator review?

---

## Architecture

```text
Transaction Data
      ↓
Data Validation & Preprocessing
      ↓
Point-in-Time Feature Engineering
      ↓
XGBoost Fraud Model
      ↓
Fraud Probability
      ↓
Risk Scoring & Alert Generation
      ↓
SHAP Explainability
      ↓
Investigation Evidence Engine
      ↓
GenAI Investigation Assistant
      ↓
Streamlit Investigation Dashboard