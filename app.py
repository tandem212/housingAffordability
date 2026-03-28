import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

st.set_page_config(page_title="Housing Stress MLP", layout="wide")


@st.cache_data
def load_data():
    results = pd.read_csv("outputs/mlp_results.csv")
    sarima = pd.read_csv("outputs/sarima_results.csv")
    return results, sarima


results, sarima = load_data()

# Compute metrics from data
y_true = results["true_label"]
y_pred = results["pred_label"]
y_prob = results["pred_prob"]

accuracy = accuracy_score(y_true, y_pred)
report = classification_report(y_true, y_pred, output_dict=True)
hs_precision = report["1"]["precision"]
hs_recall = report["1"]["recall"]
macro_f1 = report["macro avg"]["f1-score"]

# Tabs
tab1, tab2, tab3 = st.tabs(["About", "MLP Results", "SARIMA Results"])

# About
with tab1:
    st.title("Housing Affordability Stress in Greater Boston")
    st.markdown("""
    ## Overview
    This project predicts housing affordability stress across Greater Boston ZIP
    codes using Zillow home value and rent data combined with FRED macroeconomic
    indicators. A ZIP code is considered high affordability stress in a given month if the
    income needed to buy a home there puts it in the top 25% most expensive
    ZIP-months across the dataset.

    ## Models
    - **SARIMAX** (Python) — forecasts ZIP-level home values using mortgage rates
      as an exogenous variable. Fitted on six representative ZIP codes spanning all
      location tiers and price ranges.
    - **Bayesian Regression** (R) — predicts income needed to afford housing using
      mortgage rates, unemployment, CPI, inventory, and location tier as features.
    - **MLP Classifier** (Python) — manually implemented neural network that
      classifies ZIP-months as high or low affordability stress.

    ## Data Sources
    - Zillow ZHVI and ZORI at ZIP code level
    - Zillow metro-level inventory, days-to-pending, and income needed
    - FRED: 30-year mortgage rate, national and Boston unemployment, CPI

    """)

    st.subheader("MLP Test Set Performance")
    st.table(
        pd.DataFrame(
            {
                "Metric": [
                    "Accuracy",
                    "High Stress Recall",
                    "High Stress Precision",
                    "Macro F1",
                ],
                "Value": [
                    f"{accuracy:.1%}",
                    f"{hs_recall:.1%}",
                    f"{hs_precision:.1%}",
                    f"{macro_f1:.2f}",
                ],
            }
        )
    )

# MLP Results
with tab2:
    st.title("MLP Classifier Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy:.1%}")
    col2.metric("High Stress Recall", f"{hs_recall:.1%}")
    col3.metric("High Stress Precision", f"{hs_precision:.1%}")
    col4.metric("Macro F1", f"{macro_f1:.2f}")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Training Loss Curve")
        st.image("outputs/plots/mlp_loss.png")
    with col2:
        st.subheader("Predicted Probability Distribution")
        st.image("outputs/plots/mlp_hist.png")

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Confusion Matrix")
        st.image("outputs/plots/mlp_confusion_matrix.png")
    with col2:
        st.subheader("ROC Curve")
        st.image("outputs/plots/mlp_roc_curve.png")

    st.divider()

    st.subheader("Precision-Recall Curve")
    st.image("outputs/plots/mlp_precision_recall.png")

    st.divider()

    st.subheader("Predictions Explorer")
    filter_label = st.selectbox(
        "Filter by true label", ["All", "High Stress (1)", "Low Stress (0)"]
    )
    if filter_label == "High Stress (1)":
        st.dataframe(results[results["true_label"] == 1])
    elif filter_label == "Low Stress (0)":
        st.dataframe(results[results["true_label"] == 0])
    else:
        st.dataframe(results)

# SARIMA Results
with tab3:
    st.title("SARIMA Forecast Results")
    st.markdown("""
    SARIMAX(1,1,1)(1,1,1)[12] trained on 6 representative ZIP codes through end of 2022,
    forecasting 2023–2024 with 30-year mortgage rate as exogenous variable.
    """)

    st.subheader("Forecast vs Actual (2023–2024)")
    st.image("outputs/plots/sarima_forecast_all_zips.png")

    st.divider()

    st.subheader("Error Metrics by ZIP Code")
    st.dataframe(
        sarima[["label", "mae", "rmse", "mape_pct"]].rename(
            columns={
                "label": "ZIP / Neighborhood",
                "mae": "MAE ($)",
                "rmse": "RMSE ($)",
                "mape_pct": "MAPE (%)",
            }
        ),
        hide_index=True,
    )

    st.divider()

    st.subheader("ACF / PACF Analysis")
    zip_labels = {
        "02126": "Mattapan (02126)",
        "02116": "Back Bay (02116)",
        "02150": "Chelsea (02150)",
        "02139": "Cambridge (02139)",
        "01840": "Lawrence (01840)",
        "02481": "Wellesley (02481)",
    }
    selected = st.selectbox("Select ZIP code", list(zip_labels.values()))
    zip_code = [k for k, v in zip_labels.items() if v == selected][0]
    st.image(f"outputs/acf_pacf/acf_pacf_{zip_code}.png")
