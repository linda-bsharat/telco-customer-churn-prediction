# ============================================================
#  Telco Customer Churn Predictor — Streamlit App
#  Pages: Overview | Predictor | Insights | Model | About
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model    import LogisticRegression
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (accuracy_score, precision_score,
                                     recall_score, f1_score,
                                     confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────
#  Page config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title = "Telco Churn Predictor",
    page_icon  = "📡",
    layout     = "wide",
)

# ──────────────────────────────────────────────
#  Colours
# ──────────────────────────────────────────────
C_PRIMARY = "#534AB7"
C_DANGER  = "#D85A30"
C_SUCCESS = "#1D9E75"

# ──────────────────────────────────────────────
#  Global CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

[data-testid="stSidebar"] {
    background: #FAFAFA;
    border-right: 1px solid #EBEBEB;
}
[data-testid="metric-container"] {
    background: #F7F7F9;
    border: 0.5px solid #E4E4E8;
    border-radius: 10px;
    padding: 14px 18px;
}
.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 1.4px;
    text-transform: uppercase;
    color: #534AB7;
    margin-bottom: 6px;
}
.card-churn {
    background: #FEF2EE;
    border: 1.5px solid #D85A30;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
.card-safe {
    background: #EDFBF5;
    border: 1.5px solid #1D9E75;
    border-radius: 14px;
    padding: 28px;
    text-align: center;
}
div.stButton > button {
    background: #534AB7 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 9px !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 12px 36px !important;
    width: 100% !important;
}
div.stButton > button:hover { background: #3C3489 !important; }
hr { border-color: #EBEBEB !important; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  Data & Model  (cached — load once)
# ──────────────────────────────────────────────
DATA_URL = (
    "https://raw.githubusercontent.com/linda-bsharat/"
    "telco-customer-churn-prediction/refs/heads/main/"
    "data/telco_customer_data_cleaned.csv"
)

@st.cache_data(show_spinner="Loading dataset…")
def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_URL)


@st.cache_resource(show_spinner="Training model…")
def train_model(df: pd.DataFrame):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  LogisticRegression(max_iter=1000, random_state=42)),
    ])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    metrics = {
        "accuracy":  round(accuracy_score (y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score   (y_test, y_pred), 4),
        "f1":        round(f1_score       (y_test, y_pred), 4),
        "cm":        confusion_matrix(y_test, y_pred),
        "report":    classification_report(y_test, y_pred),
    }
    return pipe, X.columns.tolist(), metrics


try:
    df = load_data()
    pipeline, feature_cols, metrics = train_model(df)
except Exception as e:
    st.error(f"Could not load data: {e}")
    st.stop()


# ──────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────
def style_fig(fig, axes):
    """Apply clean white style to any figure."""
    for ax in (np.array(axes).flatten() if not isinstance(axes, plt.Axes) else [axes]):
        ax.set_facecolor("#FFFFFF")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#E4E4E8")
        ax.tick_params(colors="#666")
        ax.xaxis.label.set_color("#666")
        ax.yaxis.label.set_color("#666")
        ax.title.set_color("#1a1a1a")
    fig.patch.set_facecolor("#FFFFFF")
    return fig


def build_input_row(inputs: dict) -> pd.DataFrame:
    """Build one-row DataFrame aligned to training columns."""
    row = {}

    # Numeric
    row["SeniorCitizen"]  = 1 if inputs["senior"] == "Yes" else 0
    row["tenure"]         = inputs["tenure"]
    row["MonthlyCharges"] = inputs["monthly"]
    row["TotalCharges"]   = inputs["total_charges"]

    # Binary Yes/No
    for col, key in [("Partner","partner"), ("Dependents","dependents"),
                     ("PhoneService","phone_service"), ("PaperlessBilling","paperless")]:
        row[col] = 1 if inputs[key] == "Yes" else 0

    # Gender
    row["gender_Male"] = 1 if inputs["gender"] == "Male" else 0

    # MultipleLines
    row["MultipleLines_No phone service"] = 1 if inputs["multi_lines"] == "No phone service" else 0
    row["MultipleLines_Yes"]              = 1 if inputs["multi_lines"] == "Yes" else 0

    # InternetService
    row["InternetService_Fiber optic"] = 1 if inputs["internet"] == "Fiber optic" else 0
    row["InternetService_No"]          = 1 if inputs["internet"] == "No" else 0

    # Service add-ons
    for feat, key in [("OnlineSecurity","online_sec"), ("OnlineBackup","online_bkp"),
                      ("DeviceProtection","device_prot"), ("TechSupport","tech_support"),
                      ("StreamingTV","tv"), ("StreamingMovies","movies")]:
        row[f"{feat}_No internet service"] = 1 if inputs[key] == "No internet service" else 0
        row[f"{feat}_Yes"]                  = 1 if inputs[key] == "Yes" else 0

    # Contract
    row["Contract_One year"] = 1 if inputs["contract"] == "One year" else 0
    row["Contract_Two year"] = 1 if inputs["contract"] == "Two year" else 0

    # Payment
    row["PaymentMethod_Credit card (automatic)"] = 1 if inputs["payment"] == "Credit card (automatic)" else 0
    row["PaymentMethod_Electronic check"]        = 1 if inputs["payment"] == "Electronic check" else 0
    row["PaymentMethod_Mailed check"]            = 1 if inputs["payment"] == "Mailed check" else 0

    return pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)


# ──────────────────────────────────────────────
#  Sidebar navigation
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Telco Churn")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠  Overview", "🔮  Predictor", "📊  Insights", "🤖  Model", "ℹ️  About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Capstone Project · Logistic Regression")


# ══════════════════════════════════════════════════════════════
#  PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════
if page == "🏠  Overview":

    st.title("📡 Telco Customer Churn Predictor")
    st.markdown(
        "A machine learning system that identifies customers at risk of leaving — "
        "helping telecom teams act **before** it's too late."
    )
    st.markdown("---")

    # KPI row
    total      = len(df)
    churned    = int(df["Churn"].sum())
    retained   = total - churned
    churn_rate = churned / total * 100

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Customers", f"{total:,}")
    k2.metric("Churned",         f"{churned:,}",
              delta=f"{churn_rate:.1f}% churn rate", delta_color="inverse")
    k3.metric("Retained",        f"{retained:,}")
    k4.metric("Model F1 Score",  f"{metrics['f1']*100:.1f}%")

    st.markdown("---")

    # Row 1 charts
    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<p class="section-label">Churn distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        wedges, _ = ax.pie(
            [retained, churned],
            colors=[C_PRIMARY, C_DANGER],
            startangle=90,
            wedgeprops=dict(width=0.55, edgecolor="white", linewidth=2),
        )
        ax.legend(wedges,
                  [f"Retained  {retained/total*100:.1f}%", f"Churned  {churn_rate:.1f}%"],
                  loc="center left", bbox_to_anchor=(0.82, 0.5), fontsize=9, frameon=False)
        ax.set_title("Churn vs Retained", fontsize=12, pad=12)
        style_fig(fig, ax)
        st.pyplot(fig, use_container_width=True)

    with c2:
        st.markdown('<p class="section-label">Churn by contract type</p>', unsafe_allow_html=True)
        if "Contract" in df.columns:
            data = df.groupby("Contract")["Churn"].mean().mul(100).sort_values()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            bars = ax.barh(data.index, data.values,
                           color=[C_SUCCESS, C_PRIMARY, C_DANGER], height=0.5, edgecolor="none")
            ax.bar_label(bars, fmt="%.1f%%", padding=4, color="#444", fontsize=9)
            ax.set_xlabel("Churn Rate (%)")
            ax.set_xlim(0, data.max() * 1.25)
            ax.set_title("Churn Rate by Contract", fontsize=12)
            style_fig(fig, ax)
            st.pyplot(fig, use_container_width=True)

    # Row 2 charts
    c3, c4 = st.columns(2)

    with c3:
        st.markdown('<p class="section-label">Churn by internet service</p>', unsafe_allow_html=True)
        if "InternetService" in df.columns:
            data = df.groupby("InternetService")["Churn"].mean().mul(100).sort_values()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.barh(data.index, data.values,
                    color=[C_SUCCESS, C_PRIMARY, C_DANGER], height=0.45, edgecolor="none")
            ax.set_xlabel("Churn Rate (%)")
            ax.set_title("Churn Rate by Internet Service", fontsize=12)
            style_fig(fig, ax)
            st.pyplot(fig, use_container_width=True)

    with c4:
        st.markdown('<p class="section-label">Tenure distribution by churn</p>', unsafe_allow_html=True)
        if "tenure" in df.columns:
            fig, ax = plt.subplots(figsize=(5, 3))
            df[df["Churn"] == 0]["tenure"].plot.hist(
                ax=ax, bins=30, alpha=0.6, color=C_PRIMARY, label="Retained", edgecolor="none")
            df[df["Churn"] == 1]["tenure"].plot.hist(
                ax=ax, bins=30, alpha=0.6, color=C_DANGER,  label="Churned",  edgecolor="none")
            ax.set_xlabel("Tenure (months)")
            ax.set_ylabel("Count")
            ax.set_title("Tenure Distribution", fontsize=12)
            ax.legend(fontsize=9, frameon=False)
            style_fig(fig, ax)
            st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 2 — Predictor
# ══════════════════════════════════════════════════════════════
elif page == "🔮  Predictor":

    st.title("🔮 Customer Churn Predictor")
    st.markdown("Fill in the customer details and click **Predict** to see churn risk.")
    st.markdown("---")

    left, right = st.columns([1.3, 1], gap="large")

    with left:
        # Demographics
        st.markdown('<p class="section-label">Demographics</p>', unsafe_allow_html=True)
        d1, d2, d3 = st.columns(3)
        gender     = d1.selectbox("Gender",         ["Male", "Female"])
        senior     = d2.selectbox("Senior Citizen",  ["No", "Yes"])
        partner    = d3.selectbox("Partner",          ["No", "Yes"])
        dependents = st.selectbox("Dependents",       ["No", "Yes"])

        # Account info
        st.markdown('<p class="section-label" style="margin-top:16px">Account Info</p>',
                    unsafe_allow_html=True)
        a1, a2     = st.columns(2)
        tenure     = a1.number_input("Tenure (months)",       0, 72, 12)
        monthly    = a2.number_input("Monthly Charges ($)",   0.0, 200.0, 65.0, step=0.5)
        b1, b2     = st.columns(2)
        total_charges = b1.number_input("Total Charges ($)",  0.0, 10000.0,
                                         value=round(float(monthly * tenure), 2), step=1.0)
        contract   = b2.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        c1_, c2_   = st.columns(2)
        payment    = c1_.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ])
        paperless  = c2_.selectbox("Paperless Billing", ["No", "Yes"])

        # Services
        st.markdown('<p class="section-label" style="margin-top:16px">Services</p>',
                    unsafe_allow_html=True)
        s1, s2, s3   = st.columns(3)
        phone_service = s1.selectbox("Phone Service",    ["No", "Yes"])
        multi_lines   = s2.selectbox("Multiple Lines",   ["No", "Yes", "No phone service"])
        internet      = s3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

        t1, t2, t3   = st.columns(3)
        online_sec   = t1.selectbox("Online Security",   ["No", "Yes", "No internet service"])
        online_bkp   = t2.selectbox("Online Backup",     ["No", "Yes", "No internet service"])
        device_prot  = t3.selectbox("Device Protection", ["No", "Yes", "No internet service"])

        u1, u2, u3   = st.columns(3)
        tech_support  = u1.selectbox("Tech Support",     ["No", "Yes", "No internet service"])
        tv            = u2.selectbox("Streaming TV",     ["No", "Yes", "No internet service"])
        movies        = u3.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

        st.markdown("")
        predict_btn = st.button("🔮 Predict Churn Risk")

    with right:
        st.markdown('<p class="section-label">Prediction Result</p>', unsafe_allow_html=True)

        if predict_btn:
            inputs = dict(
                gender=gender, senior=senior, partner=partner, dependents=dependents,
                tenure=tenure, contract=contract, payment=payment, paperless=paperless,
                phone_service=phone_service, multi_lines=multi_lines, internet=internet,
                online_sec=online_sec, online_bkp=online_bkp, device_prot=device_prot,
                tech_support=tech_support, tv=tv, movies=movies,
                monthly=monthly, total_charges=total_charges,
            )
            input_df   = build_input_row(inputs)
            pred       = pipeline.predict(input_df)[0]
            prob       = pipeline.predict_proba(input_df)[0]
            churn_prob = prob[1] * 100
            stay_prob  = prob[0] * 100

            if pred == 1:
                st.markdown(f"""
                <div class="card-churn">
                    <div style="font-size:2.8rem">⚠️</div>
                    <div style="font-size:1.5rem;font-weight:600;color:#712B13;margin:8px 0">High Risk</div>
                    <div style="font-size:2.6rem;font-weight:700;color:#D85A30">{churn_prob:.1f}%</div>
                    <div style="color:#993C1D;font-size:0.9rem;margin-top:4px">Churn Probability</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="card-safe">
                    <div style="font-size:2.8rem">✅</div>
                    <div style="font-size:1.5rem;font-weight:600;color:#085041;margin:8px 0">Low Risk</div>
                    <div style="font-size:2.6rem;font-weight:700;color:#1D9E75">{stay_prob:.1f}%</div>
                    <div style="color:#0F6E56;font-size:0.9rem;margin-top:4px">Retention Probability</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability bar
            fig, ax = plt.subplots(figsize=(4.5, 2))
            bars = ax.barh(["Churn", "Stay"], [churn_prob, stay_prob],
                           color=[C_DANGER, C_SUCCESS], height=0.4, edgecolor="none")
            ax.bar_label(bars, fmt="%.1f%%", padding=4, color="#444", fontsize=10)
            ax.set_xlim(0, 115)
            ax.set_title("Probability Breakdown", fontsize=11)
            style_fig(fig, ax)
            st.pyplot(fig, use_container_width=True)

            # Quick summary
            st.markdown("**Input summary:**")
            st.table(pd.DataFrame({
                "Field": ["Contract", "Tenure", "Monthly Charges", "Internet", "Tech Support"],
                "Value": [contract, f"{tenure} months", f"${monthly:.2f}", internet, tech_support],
            }).set_index("Field"))

        else:
            st.markdown("""
            <div style="border:1.5px dashed #DDDDE8;border-radius:14px;
                        padding:60px 20px;text-align:center;color:#AAA">
                <div style="font-size:3rem;margin-bottom:10px">🔮</div>
                <div style="font-size:0.95rem">Fill the form and click<br>
                    <b style="color:#534AB7">Predict Churn Risk</b>
                </div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 3 — Insights
# ══════════════════════════════════════════════════════════════
elif page == "📊  Insights":

    st.title("📊 Data Insights")
    st.markdown("Key patterns and drivers discovered during exploratory data analysis.")
    st.markdown("---")

    # Feature importance
    st.markdown('<p class="section-label">Top Churn Drivers — Logistic Regression Coefficients</p>',
                unsafe_allow_html=True)

    coefs   = pd.Series(pipeline.named_steps["model"].coef_[0], index=feature_cols)
    top_all = pd.concat([coefs.nsmallest(8), coefs.nlargest(8)]).sort_values()

    fig, ax = plt.subplots(figsize=(9, 5))
    colors  = [C_DANGER if v > 0 else C_SUCCESS for v in top_all.values]
    ax.barh(top_all.index, top_all.values, color=colors, height=0.6, edgecolor="none")
    ax.axvline(0, color="#CCCCCC", linewidth=0.8)
    ax.set_xlabel("Coefficient Value  (positive = increases churn risk)")
    ax.set_title("Feature Coefficients — Logistic Regression", fontsize=13, pad=12)
    ax.legend(handles=[
        mpatches.Patch(color=C_DANGER,  label="Increases churn risk"),
        mpatches.Patch(color=C_SUCCESS, label="Reduces churn risk"),
    ], fontsize=9, frameon=False)
    style_fig(fig, ax)
    st.pyplot(fig, use_container_width=True)

    st.markdown("---")

    # Row 1
    i1, i2 = st.columns(2)

    with i1:
        st.markdown('<p class="section-label">Monthly charges distribution</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 3.5))
        df[df["Churn"] == 0]["MonthlyCharges"].plot.hist(
            ax=ax, bins=30, alpha=0.6, color=C_PRIMARY, label="Retained", edgecolor="none")
        df[df["Churn"] == 1]["MonthlyCharges"].plot.hist(
            ax=ax, bins=30, alpha=0.6, color=C_DANGER,  label="Churned",  edgecolor="none")
        ax.set_xlabel("Monthly Charges ($)")
        ax.set_ylabel("Count")
        ax.set_title("Monthly Charges by Churn", fontsize=12)
        ax.legend(fontsize=9, frameon=False)
        style_fig(fig, ax)
        st.pyplot(fig, use_container_width=True)

    with i2:
        st.markdown('<p class="section-label">Churn rate by payment method</p>', unsafe_allow_html=True)
        if "PaymentMethod" in df.columns:
            data = df.groupby("PaymentMethod")["Churn"].mean().mul(100).sort_values()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            colors_ = [C_DANGER if v == data.max() else C_PRIMARY for v in data.values]
            bars = ax.barh(data.index, data.values,
                           color=colors_, height=0.45, edgecolor="none")
            ax.bar_label(bars, fmt="%.1f%%", padding=3, color="#444", fontsize=9)
            ax.set_xlabel("Churn Rate (%)")
            ax.set_xlim(0, data.max() * 1.25)
            ax.set_title("Churn by Payment Method", fontsize=12)
            style_fig(fig, ax)
            st.pyplot(fig, use_container_width=True)

    # Row 2
    i3, i4 = st.columns(2)

    with i3:
        st.markdown('<p class="section-label">Senior citizen churn rate</p>', unsafe_allow_html=True)
        if "SeniorCitizen" in df.columns:
            data = df.groupby("SeniorCitizen")["Churn"].mean().mul(100)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(["Non-Senior", "Senior"], data.values,
                   color=[C_PRIMARY, C_DANGER], width=0.4, edgecolor="none")
            for i, v in enumerate(data.values):
                ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=10, color="#444")
            ax.set_ylabel("Churn Rate (%)")
            ax.set_title("Churn: Senior vs Non-Senior", fontsize=12)
            style_fig(fig, ax)
            st.pyplot(fig, use_container_width=True)

    with i4:
        st.markdown('<p class="section-label">Churn rate by gender</p>', unsafe_allow_html=True)
        if "gender" in df.columns:
            data = df.groupby("gender")["Churn"].mean().mul(100).sort_values()
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(data.index, data.values,
                   color=[C_PRIMARY, "#AFA9EC"], width=0.35, edgecolor="none")
            for i, v in enumerate(data.values):
                ax.text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=10, color="#444")
            ax.set_ylabel("Churn Rate (%)")
            ax.set_ylim(0, data.max() * 1.3)
            ax.set_title("Churn Rate by Gender", fontsize=12)
            style_fig(fig, ax)
            st.pyplot(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
#  PAGE 4 — Model
# ══════════════════════════════════════════════════════════════
elif page == "🤖  Model":

    st.title("🤖 Model Performance")
    st.markdown("Evaluation of the Logistic Regression model on the held-out test set.")
    st.markdown("---")

    # KPI row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy",  f"{metrics['accuracy']*100:.2f}%")
    m2.metric("Precision", f"{metrics['precision']*100:.2f}%")
    m3.metric("Recall",    f"{metrics['recall']*100:.2f}%")
    m4.metric("F1 Score",  f"{metrics['f1']*100:.2f}%")

    st.markdown("---")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown('<p class="section-label">Confusion Matrix</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            metrics["cm"], annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            linewidths=0.5, linecolor="#E4E4E8", ax=ax,
        )
        ax.set_xlabel("Predicted Label", labelpad=10)
        ax.set_ylabel("True Label",      labelpad=10)
        ax.set_title("Confusion Matrix — Test Set", fontsize=12, pad=12)
        fig.patch.set_facecolor("#FFFFFF")
        st.pyplot(fig, use_container_width=True)

    with right:
        st.markdown('<p class="section-label">Classification Report</p>', unsafe_allow_html=True)
        st.code(metrics["report"], language=None)

        st.markdown('<p class="section-label" style="margin-top:20px">Model Comparison</p>',
                    unsafe_allow_html=True)
        st.dataframe(pd.DataFrame({
            "Model":     ["Logistic Regression ✓", "Random Forest", "Neural Network"],
            "Accuracy":  ["77.0%", "76.4%", "76.8%"],
            "Precision": ["70.1%", "70.2%", "69.7%"],
            "Recall":    ["89.6%", "86.8%", "89.5%"],
            "F1 Score":  ["78.7%", "77.6%", "78.4%"],
        }), hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown('<p class="section-label">Why Logistic Regression?</p>', unsafe_allow_html=True)
    st.info(
        "Despite similar accuracy to ensemble and deep learning models, "
        "Logistic Regression was chosen for its **interpretability** — "
        "each coefficient directly quantifies how much a feature increases "
        "or decreases churn probability, making it actionable for business teams."
    )


# ══════════════════════════════════════════════════════════════
#  PAGE 5 — About
# ══════════════════════════════════════════════════════════════
elif page == "ℹ️  About":

    st.title("ℹ️ About the Project")
    st.markdown("---")

    a1, a2 = st.columns(2, gap="large")

    with a1:
        st.markdown("### Problem Statement")
        st.write(
            "Telecom companies face significant revenue loss from customer churn. "
            "This project builds a machine learning pipeline that predicts which "
            "customers are likely to leave, enabling proactive retention strategies."
        )

        st.markdown("### Dataset")
        st.write(
            "**~70,000** customer records with **21 features** covering demographics, "
            "service subscriptions, account information, and billing details. "
            "Target variable: `Churn` (Yes / No)."
        )

        st.markdown("### Methodology")
        for i, step in enumerate([
            "Data Cleaning & Handling Missing Values",
            "Exploratory Data Analysis (EDA)",
            "Feature Engineering & Encoding",
            "Model Training — LR · Random Forest · Neural Network",
            "Evaluation & Model Selection",
            "Deployment via Streamlit",
        ], 1):
            st.markdown(f"**{i}.** {step}")

    with a2:
        st.markdown("### Tools & Libraries")
        for k, v in {
            "Language":        "Python 3.11",
            "Data":            "Pandas · NumPy",
            "ML":              "Scikit-learn · XGBoost",
            "Deep Learning":   "TensorFlow / Keras",
            "Visualization":   "Matplotlib · Seaborn",
            "App":             "Streamlit",
            "Version Control": "Git · GitHub",
        }.items():
            st.markdown(f"**{k}:** {v}")

        st.markdown("### Repository")
        st.markdown(
            "🔗 [github.com/linda-bsharat/telco-customer-churn-prediction]"
            "(https://github.com/linda-bsharat/telco-customer-churn-prediction)"
        )

        st.markdown("### Key Findings")
        st.success(
            "• Month-to-month contracts drive **89%** of churn cases\n\n"
            "• Fiber optic customers churn at nearly **3×** the rate of DSL users\n\n"
            "• Customers with tenure < 12 months are the **highest risk** group\n\n"
            "• Electronic check payment correlates strongly with higher churn"
        )