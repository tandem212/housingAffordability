from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
PLOTS = OUTPUTS / "plots"
ACF_PACF_DIR = OUTPUTS / "acf_pacf"

st.set_page_config(
    page_title="Boston Housing Affordability",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def money(x: float) -> str:
    return f"${x:,.0f}"


def show_image(path: Path) -> None:
    if path.exists():
        st.image(str(path), use_container_width=True)
    else:
        st.warning(f"Missing image: {path.name}")


@st.cache_data(show_spinner=False)
def load_data() -> dict[str, pd.DataFrame]:
    return {
        "mlp": pd.read_csv(OUTPUTS / "mlp_results.csv"),
        "sarima": pd.read_csv(OUTPUTS / "sarima_results.csv"),
        "bayes_preds": pd.read_csv(OUTPUTS / "bayesian_predictions.csv", parse_dates=["date"]),
        "bayes_metrics": pd.read_csv(OUTPUTS / "bayesian_metrics.csv"),
    }


data = load_data()
mlp = data["mlp"].copy()
sarima = data["sarima"].copy()
bayes_preds = data["bayes_preds"].copy()
bayes_metrics = data["bayes_metrics"].copy()

mlp["true_label"] = mlp["true_label"].astype(int)
mlp["pred_label"] = mlp["pred_label"].astype(int)
mlp["pred_prob"] = mlp["pred_prob"].astype(float)

bayes_preds["zip_code"] = bayes_preds["zip_code"].astype(str).str.zfill(5)

y_true = mlp["true_label"]
y_pred = mlp["pred_label"]
y_prob = mlp["pred_prob"]

mlp_accuracy = accuracy_score(y_true, y_pred)
mlp_report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
mlp_recall = mlp_report["1"]["recall"]

bayes_row = bayes_metrics.iloc[0]
bayes_rmse = float(bayes_row["rmse"])
bayes_mae = float(bayes_row["mae"])
bayes_mape = float(bayes_row["mape_pct"])


sarima["mape_pct"] = sarima["mape_pct"].astype(float)
best_sarima = sarima.sort_values("mape_pct").iloc[0]
worst_sarima = sarima.sort_values("mape_pct", ascending=False).iloc[0]


def render_home() -> None:
    st.title("Housing Affordability Stress in Greater Boston")
    st.markdown(
        """
        Housing affordability can change quickly, and early warning signs are easy to miss.
        This project combines housing and economic signals to answer a simple question:
        **which neighborhoods are becoming harder to afford, and how fast?**

        ### Explore the story from three angles
        - **Bayesian Model**: estimates the yearly income a household would need to afford a typical home.
        - **MLP Results**: labels ZIP-months as higher-stress or lower-stress based on market conditions.
        - **SARIMA Results**: forecasts how home values may move next, ZIP by ZIP.

        ### Why this matters?
        Affordability stress is not uniform. Two nearby neighborhoods can move in very different directions.
        By combining these methods, the app helps you see both:
        - the **current pressure** (how much income is needed now), and
        - the **near-term direction** (where stress may go next).
        """
    )

    st.subheader("Quick visual guide")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("🏠 **Affordability pressure**\n\nHow expensive is it to own a typical home in each ZIP?")
    with c2:
        st.info("📍 **Stress map by ZIP-month**\n\nWhich places are moving into high-stress conditions?")
    with c3:
        st.info("📈 **What may happen next**\n\nAre local home values likely to cool, hold, or keep rising?")


def render_bayesian() -> None:
    st.title("Bayesian Model")
    st.write("Predicts annual income needed using macro and housing market features.")

    c1, c2, c3 = st.columns(3)
    c1.metric("RMSE", money(bayes_rmse))
    c2.metric("MAE", money(bayes_mae))
    c3.metric("MAPE", f"{bayes_mape:.2f}%")

    st.subheader("Posterior coefficient intervals")
    show_image(PLOTS / "bayesian_coef_intervals.png")

    st.subheader("Model fit over time")
    st.caption("These two plots use the full test-period predictions.")

    trend = (
        bayes_preds.assign(abs_residual=bayes_preds["residual"].abs())
        .groupby("date", as_index=False)
        .agg(
            actual_income_needed=("actual_income_needed", "mean"),
            predicted_income_needed=("predicted_income_needed", "mean"),
            abs_residual=("abs_residual", "mean"),
        )
    )

    t1, t2 = st.columns(2)
    with t1:
        st.subheader("Monthly average income needed: actual vs model prediction")
        st.line_chart(trend.set_index("date")[["actual_income_needed", "predicted_income_needed"]])
    with t2:
        st.subheader("Average absolute residual over time")
        st.line_chart(trend.set_index("date")[["abs_residual"]])


def render_mlp() -> None:
    st.title("MLP Results")
    st.write("Binary classifier for identifying high affordability stress ZIP-months.")

    st.caption(
        "Decision threshold changes the probability cutoff for classifying a row as high stress. "
        "Lower threshold increases recall but can increase false positives."
    )
    threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)

    threshold_pred = (y_prob >= threshold).astype(int)
    threshold_report = classification_report(y_true, threshold_pred, output_dict=True, zero_division=0)
    threshold_conf = confusion_matrix(y_true, threshold_pred)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y_true, threshold_pred):.1%}")
    c2.metric("Recall (class 1)", f"{threshold_report['1']['recall']:.1%}")
    c3.metric("Precision (class 1)", f"{threshold_report['1']['precision']:.1%}")
    c4.metric("Macro F1", f"{threshold_report['macro avg']['f1-score']:.2f}")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.subheader("Training loss")
        show_image(PLOTS / "mlp_loss.png")
    with p2:
        st.subheader("ROC")
        show_image(PLOTS / "mlp_roc_curve.png")
    with p3:
        st.subheader("Precision-recall")
        show_image(PLOTS / "mlp_precision_recall.png")

    a1, a2 = st.columns(2)
    with a1:
        st.subheader("Confusion matrix at current threshold")
        st.dataframe(
            pd.DataFrame(threshold_conf, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]),
            use_container_width=True,
        )
    with a2:
        st.subheader("Prediction probability distribution")
        show_image(PLOTS / "mlp_hist.png")


def render_sarima() -> None:
    st.title("SARIMA Results")
    st.write("SARIMAX forecasts ZIP-level home values using mortgage rate as an exogenous variable.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Best MAPE", f"{best_sarima['mape_pct']:.2f}%")
    c2.metric("Worst MAPE", f"{worst_sarima['mape_pct']:.2f}%")
    c3.metric("ZIPs evaluated", f"{len(sarima):,}")

    st.subheader("Forecast vs actual")
    show_image(PLOTS / "sarima_forecast_all_zips.png")

    left, right = st.columns(2)
    with left:
        st.subheader("Error summary by ZIP")
        table = (
            sarima[["label", "mae", "rmse", "mape_pct"]]
            .rename(columns={"label": "ZIP", "mae": "MAE ($)", "rmse": "RMSE ($)", "mape_pct": "MAPE (%)"})
            .sort_values("MAPE (%)")
        )
        st.dataframe(table, hide_index=True, use_container_width=True)
    with right:
        st.subheader("ACF and PACF by ZIP")
        zip_labels = {
            "02126": "Mattapan (02126)",
            "02116": "Back Bay (02116)",
            "02150": "Chelsea (02150)",
            "02139": "Cambridge (02139)",
            "01840": "Lawrence (01840)",
            "02481": "Wellesley (02481)",
        }
        selected_label = st.selectbox("Select ZIP", list(zip_labels.values()), key="sarima_zip")
        selected_zip = [k for k, v in zip_labels.items() if v == selected_label][0]
        show_image(ACF_PACF_DIR / f"acf_pacf_{selected_zip}.png")


st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Bayesian Model", "MLP Results", "SARIMA Results"])

if page == "Home":
    render_home()
elif page == "Bayesian Model":
    render_bayesian()
elif page == "MLP Results":
    render_mlp()
else:
    render_sarima()
