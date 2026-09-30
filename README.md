# 🔍 FraudLens — Real-Time Fraud Detection & Investigation Platform

> An end-to-end machine learning platform for fraud detection, explainability, investigation, and investigator workflow management.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/App-Streamlit-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Container-Docker-blue.svg)](https://www.docker.com/)

## 🚀 Live Demo

### [▶️ Open FraudLens Dashboard](https://fraudlens-xxxxx.streamlit.app)

> The live application provides fraud-risk scoring, investigation queues, SHAP explanations, investigation evidence, GenAI summaries, and investigator action tracking.

---

## 📌 Overview

FraudLens is an end-to-end fraud detection and investigation platform designed to go beyond a simple binary fraud classifier.

Traditional fraud detection systems often answer only:

> **"Is this transaction suspicious?"**

FraudLens extends this workflow by answering three practical questions:

1. **Is the transaction likely to be fraudulent?**
2. **Why does the model consider it suspicious?**
3. **What evidence should an investigator review?**

The platform combines:

- Machine Learning
- Behavioral Feature Engineering
- XGBoost
- SHAP Explainability
- Rule-Based Investigation Evidence
- Risk Scoring
- GenAI Investigation Assistance
- Investigator Workflow Management
- Data Drift Monitoring
- Streamlit
- Docker

---

# 🏗️ System Architecture

```text
                    Transaction Data
                           │
                           ▼
              ┌─────────────────────────┐
              │ Data Validation &        │
              │ Preprocessing            │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Point-in-Time Feature   │
              │ Engineering              │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ XGBoost Fraud Detection │
              │ Model                   │
              └────────────┬────────────┘
                           │
                           ▼
                  Fraud Probability
                           │
                           ▼
              ┌─────────────────────────┐
              │ Risk Scoring & Alerting │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ SHAP Explainability     │
              └────────────┬────────────┘
                           │
                           ▼
              ┌─────────────────────────┐
              │ Investigation Evidence  │
              │ Engine                  │
              └────────────┬────────────┘
                           │
                    ┌──────┴──────┐
                    ▼             ▼
             GenAI Assistant   Investigator
                    │             │
                    └──────┬──────┘
                           ▼
              ┌─────────────────────────┐
              │ Streamlit Investigation│
              │ Dashboard              │
              └─────────────────────────┘
