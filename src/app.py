import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings("ignore")
 
# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)
 
# ──────────────────────────────────────────────
# CUSTOM CSS  – dark tech aesthetic
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
 
/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0e1a;
    color: #c9d1e0;
}
.stApp { background-color: #0a0e1a; }
 
/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1220 0%, #0a0e1a 100%);
    border-right: 1px solid #1e2d4a;
}
section[data-testid="stSidebar"] * { color: #c9d1e0 !important; }
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label { color: #7a9cc6 !important; font-size: 0.78rem !important; letter-spacing: 0.05em; text-transform: uppercase; }
 
/* ── Headers ── */
h1 { font-family: 'Space Mono', monospace !important; color: #e8f0fe !important; font-size: 1.6rem !important; letter-spacing: -0.02em; }
h2 { font-family: 'Space Mono', monospace !important; color: #7ec8e3 !important; font-size: 1.05rem !important; letter-spacing: 0.08em; text-transform: uppercase; border-bottom: 1px solid #1e2d4a; padding-bottom: 6px; }
h3 { font-family: 'DM Sans', sans-serif !important; color: #a8c4e0 !important; font-size: 0.95rem !important; }
 
/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: #0f1829;
    border: 1px solid #1e2d4a;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 0 18px rgba(0,180,255,0.04);
    transition: border-color 0.2s;
}
div[data-testid="metric-container"]:hover { border-color: #3a6ea8; }
div[data-testid="metric-container"] label { color: #5a80a8 !important; font-size: 0.72rem !important; letter-spacing: 0.1em; text-transform: uppercase; }
div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #7ec8e3 !important; font-family: 'Space Mono', monospace !important; font-size: 1.6rem !important; }
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { color: #4caf84 !important; }
 
/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { gap: 2px; background: #0d1220; border-radius: 8px; padding: 4px; border: 1px solid #1e2d4a; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #5a80a8; font-size: 0.82rem; letter-spacing: 0.05em; text-transform: uppercase; border-radius: 6px; padding: 8px 18px; }
.stTabs [aria-selected="true"] { background: #1a2d4a !important; color: #7ec8e3 !important; font-weight: 600; }
 
/* ── Selectbox / Slider ── */
.stSelectbox > div > div { background: #0f1829; border: 1px solid #1e2d4a; border-radius: 8px; color: #c9d1e0; }
.stSlider > div { color: #c9d1e0; }
 
/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1a3a6a 0%, #1e4d8c 100%);
    color: #7ec8e3;
    border: 1px solid #2a5490;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    letter-spacing: 0.06em;
    padding: 10px 28px;
    transition: all 0.2s;
}
.stButton > button:hover { background: linear-gradient(135deg, #1e4d8c 0%, #2460b0 100%); border-color: #4a8adc; color: #a8d8f0; box-shadow: 0 0 20px rgba(78,154,240,0.25); }
 
/* ── Info/warning boxes ── */
.stAlert { background: #0f1829; border-left: 3px solid #7ec8e3; border-radius: 0 8px 8px 0; }
 
/* ── Prediction result card ── */
.pred-card-churn {
    background: linear-gradient(135deg, #1a0a0a 0%, #2a1010 100%);
    border: 1px solid #c0392b;
    border-radius: 12px;
    padding: 24px 28px;
    box-shadow: 0 0 30px rgba(192,57,43,0.2);
    text-align: center;
}
.pred-card-stay {
    background: linear-gradient(135deg, #0a1a0e 0%, #102a18 100%);
    border: 1px solid #27ae60;
    border-radius: 12px;
    padding: 24px 28px;
    box-shadow: 0 0 30px rgba(39,174,96,0.2);
    text-align: center;
}
.pred-title { font-family: 'Space Mono', monospace; font-size: 1.3rem; font-weight: 700; margin-bottom: 8px; }
.pred-sub   { font-size: 0.88rem; color: #8a9ab8; }
.prob-bar-outer { background: #1a2a3a; border-radius: 20px; height: 10px; margin: 12px 0 4px; overflow: hidden; }
.prob-bar-inner-churn  { background: linear-gradient(90deg, #e74c3c, #c0392b); height: 100%; border-radius: 20px; }
.prob-bar-inner-stay   { background: linear-gradient(90deg, #2ecc71, #27ae60); height: 100%; border-radius: 20px; }
 
/* ── Section cards ── */
.section-card {
    background: #0f1829;
    border: 1px solid #1e2d4a;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 18px;
}
 
/* ── Divider ── */
hr { border-color: #1e2d4a; }
 
/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #1e2d4a; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)
 
# ──────────────────────────────────────────────
# MATPLOTLIB DARK THEME
# ──────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":  "#0f1829",
    "axes.facecolor":    "#0f1829",
    "axes.edgecolor":    "#1e2d4a",
    "axes.labelcolor":   "#7a9cc6",
    "xtick.color":       "#5a7a9a",
    "ytick.color":       "#5a7a9a",
    "text.color":        "#c9d1e0",
    "grid.color":        "#1a2a3a",
    "grid.linewidth":    0.6,
    "font.family":       "sans-serif",
    "axes.titlesize":    11,
    "axes.labelsize":    9,
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
})
 
ACCENT   = "#7ec8e3"
CHURN_C  = "#e74c3c"
STAY_C   = "#2ecc71"
PALETTE  = [ACCENT, CHURN_C, STAY_C, "#f39c12", "#9b59b6", "#1abc9c"]
 
# ──────────────────────────────────────────────
# DATA & MODEL LOADING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/linda-bsharat/telco-customer-churn-prediction/refs/heads/main/data/telco_customer_data_cleaned.csv"
    try:
        df = pd.read_csv(url)
    except Exception:
        st.error("⚠️ Could not load remote dataset. Please check your internet connection.")
        st.stop()
    bool_cols = [c for c in df.columns if df[c].dtype == bool]
    for c in bool_cols:
        df[c] = df[c].astype(int)
    df = df.dropna(subset=["Churn"])
    df["Churn"] = df["Churn"].astype(int)
    return df
 
@st.cache_resource
def train_models(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    # Logistic Regression
    lr = Pipeline([("scaler", StandardScaler()),
                   ("model", LogisticRegression(max_iter=1000, random_state=42))])
    lr.fit(X_train, y_train)
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
 
    models = {"Logistic Regression": lr, "Random Forest": rf}
    metrics = {}
    for name, mdl in models.items():
        yp = mdl.predict(X_test)
        yp_prob = mdl.predict_proba(X_test)[:, 1]
        metrics[name] = {
            "Accuracy":  round(accuracy_score(y_test, yp)  * 100, 1),
            "Precision": round(precision_score(y_test, yp) * 100, 1),
            "Recall":    round(recall_score(y_test, yp)    * 100, 1),
            "F1 Score":  round(f1_score(y_test, yp)        * 100, 1),
            "AUC":       round(roc_auc_score(y_test, yp_prob) * 100, 1),
            "y_test": y_test, "y_pred": yp, "y_prob": yp_prob,
        }
    return models, metrics, X_train.columns.tolist(), X_test, y_test
 
# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:18px 0 10px; text-align:center;'>
        <div style='font-family:"Space Mono",monospace; font-size:1.1rem; color:#7ec8e3; letter-spacing:0.1em;'>📡 TELCO</div>
        <div style='font-family:"Space Mono",monospace; font-size:0.7rem; color:#3a6a9a; letter-spacing:0.2em; margin-top:2px;'>CHURN INTELLIGENCE</div>
    </div>
    <hr style='margin:8px 0 18px;'>
    """, unsafe_allow_html=True)
 
    nav = st.selectbox(
        "NAVIGATION",
        ["🏠  Overview", "📊  Data Explorer", "🤖  Model Performance", "🔮  Predict Customer"],
        label_visibility="visible"
    )
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem;color:#3a5a7a;letter-spacing:0.08em;'>DATASET</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.82rem;color:#5a80a8;'>Telco Customer Churn<br>~68K records · 24 features</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem;color:#3a5a7a;letter-spacing:0.08em;'>MODELS</p>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.82rem;color:#5a80a8;'>Logistic Regression<br>Random Forest</p>", unsafe_allow_html=True)
    st.markdown("<br><p style='font-size:0.68rem;color:#2a4060;text-align:center;'>IBT × GGateway Bootcamp · 2026</p>", unsafe_allow_html=True)
 
# ──────────────────────────────────────────────
# LOAD DATA & TRAIN
# ──────────────────────────────────────────────
with st.spinner("Loading data & training models…"):
    df = load_data()
    models, metrics, feature_cols, X_test, y_test = train_models(df)
 
# ══════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════
if nav == "🏠  Overview":
    st.markdown("# Telco Customer Churn Intelligence")
    st.markdown("<p style='color:#5a80a8;font-size:0.9rem;margin-top:-10px;margin-bottom:28px;'>Predict · Analyse · Retain</p>", unsafe_allow_html=True)
 
    churn_rate = df["Churn"].mean() * 100
    total      = len(df)
    churned    = df["Churn"].sum()
    avg_charge = df["AvgMonthlyCharge"].mean()
 
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers",  f"{total:,}")
    c2.metric("Churned",          f"{churned:,}", f"{churn_rate:.1f}%")
    c3.metric("Churn Rate",       f"{churn_rate:.1f}%")
    c4.metric("Avg Monthly Charge", f"${avg_charge:.1f}")
 
    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns([1, 1])
 
    # Churn donut
    with col_l:
        st.markdown("## Churn Distribution")
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        sizes  = [churned, total - churned]
        colors = [CHURN_C, STAY_C]
        wedge_props = dict(width=0.55, edgecolor="#0f1829", linewidth=3)
        ax.pie(sizes, colors=colors, wedgeprops=wedge_props, startangle=90)
        ax.text(0, 0, f"{churn_rate:.1f}%\nChurn", ha="center", va="center",
                fontsize=13, fontweight="bold", color="#e8f0fe",
                fontfamily="monospace")
        legend_patches = [
            mpatches.Patch(color=CHURN_C, label=f"Churned  ({churned:,})"),
            mpatches.Patch(color=STAY_C,  label=f"Retained ({total-churned:,})"),
        ]
        ax.legend(handles=legend_patches, loc="lower center",
                  frameon=False, fontsize=8, labelcolor="#c9d1e0",
                  bbox_to_anchor=(0.5, -0.06), ncol=2)
        fig.patch.set_facecolor("#0f1829")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    # Contract type churn
    with col_r:
        st.markdown("## Churn Rate by Contract")
        contract_map = {0: "Month-to-Month", 1: "One Year", 2: "Two Year"}
        df_c = df.copy()
        df_c["ContractName"] = df_c["Contract"].map(contract_map)
        rates = df_c.groupby("ContractName")["Churn"].mean() * 100
 
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        bars = ax.barh(rates.index, rates.values,
                       color=[CHURN_C, ACCENT, STAY_C],
                       height=0.5, edgecolor="none")
        for bar, val in zip(bars, rates.values):
            ax.text(val + 0.8, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=9, color="#e8f0fe")
        ax.set_xlim(0, max(rates.values) * 1.22)
        ax.set_xlabel("Churn Rate (%)")
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right", "left"]].set_visible(False)
        fig.patch.set_facecolor("#0f1829")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    # Key Insights
    st.markdown("## Key Insights")
    st.markdown("""
    <div class="section-card">
    <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:18px;">
        <div>
            <p style="color:#7ec8e3;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;">📋 Contract Type</p>
            <p style="font-size:0.88rem;color:#c9d1e0;">Month-to-month customers churn at a significantly higher rate than those on annual or two-year contracts.</p>
        </div>
        <div>
            <p style="color:#f39c12;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;">⏱️ Tenure Effect</p>
            <p style="font-size:0.88rem;color:#c9d1e0;">New customers (tenure ≤ 6 months) are at higher risk. Loyalty sharply increases after the first year.</p>
        </div>
        <div>
            <p style="color:#2ecc71;font-size:0.75rem;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:6px;">🌐 Internet Service</p>
            <p style="font-size:0.88rem;color:#c9d1e0;">Fiber Optic subscribers show slightly higher churn — possibly due to higher costs or competitive alternatives.</p>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)
 
# ══════════════════════════════════════════════
# PAGE: DATA EXPLORER
# ══════════════════════════════════════════════
elif nav == "📊  Data Explorer":
    st.markdown("# Data Explorer")
    st.markdown("<p style='color:#5a80a8;font-size:0.9rem;margin-top:-10px;margin-bottom:24px;'>Explore patterns in the cleaned dataset</p>", unsafe_allow_html=True)
 
    tab1, tab2, tab3 = st.tabs(["DISTRIBUTIONS", "CORRELATIONS", "FEATURE DEEP DIVE"])
 
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Tenure Distribution by Churn")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for label, color in [(0, STAY_C), (1, CHURN_C)]:
                subset = df[df["Churn"] == label]["tenure"]
                ax.hist(subset, bins=30, alpha=0.6, color=color,
                        label="Retained" if label == 0 else "Churned",
                        edgecolor="none", density=True)
            ax.set_xlabel("Tenure (months)")
            ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            fig.patch.set_facecolor("#0f1829")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
 
        with col2:
            st.markdown("### Avg Monthly Charge by Churn")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            for label, color in [(0, STAY_C), (1, CHURN_C)]:
                subset = df[df["Churn"] == label]["AvgMonthlyCharge"]
                ax.hist(subset, bins=30, alpha=0.6, color=color,
                        label="Retained" if label == 0 else "Churned",
                        edgecolor="none", density=True)
            ax.set_xlabel("Avg Monthly Charge ($)")
            ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=8)
            ax.spines[["top", "right"]].set_visible(False)
            fig.patch.set_facecolor("#0f1829")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
 
        col3, col4 = st.columns(2)
        with col3:
            st.markdown("### Churn by Internet Service")
            inet_map = {0: "No Internet", 1: "DSL", 2: "Fiber Optic"}
            df["InternetName"] = df["InternetService"].map(inet_map)
            rates = df.groupby("InternetName")["Churn"].mean() * 100
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.bar(rates.index, rates.values,
                   color=[ACCENT, "#f39c12", CHURN_C], edgecolor="none", width=0.5)
            for i, v in enumerate(rates.values):
                ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9, color="#e8f0fe")
            ax.set_ylabel("Churn Rate (%)")
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.grid(axis="y", alpha=0.3)
            fig.patch.set_facecolor("#0f1829")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
 
        with col4:
            st.markdown("### Churn by Total Services Used")
            rates2 = df.groupby("TotalServices")["Churn"].mean() * 100
            fig, ax = plt.subplots(figsize=(5, 3.5))
            ax.plot(rates2.index, rates2.values, color=ACCENT,
                    linewidth=2.5, marker="o", markersize=6,
                    markerfacecolor=CHURN_C, markeredgecolor="#0f1829")
            ax.fill_between(rates2.index, rates2.values, alpha=0.12, color=ACCENT)
            ax.set_xlabel("Number of Services")
            ax.set_ylabel("Churn Rate (%)")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(alpha=0.25)
            fig.patch.set_facecolor("#0f1829")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
 
    with tab2:
        st.markdown("### Feature Correlation Heatmap")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(13, 9))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap="coolwarm", center=0,
                    annot=True, fmt=".2f", annot_kws={"size": 6.5},
                    linewidths=0.4, linecolor="#0a0e1a",
                    cbar_kws={"shrink": 0.6}, ax=ax)
        ax.tick_params(labelsize=7.5, rotation=45)
        fig.patch.set_facecolor("#0f1829")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    with tab3:
        st.markdown("### Explore Any Feature vs Churn")
        feat = st.selectbox(
            "Select feature",
            [c for c in df.select_dtypes(include=[np.number]).columns if c not in ["Churn", "InternetName"]],
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        # Distribution
        for label, color in [(0, STAY_C), (1, CHURN_C)]:
            axes[0].hist(df[df["Churn"] == label][feat], bins=25,
                         alpha=0.65, color=color, edgecolor="none", density=True,
                         label="Retained" if label == 0 else "Churned")
        axes[0].set_title(f"Distribution — {feat}")
        axes[0].set_xlabel(feat)
        axes[0].legend(frameon=False, fontsize=8)
        axes[0].spines[["top", "right"]].set_visible(False)
        # Box
        data_stay  = df[df["Churn"] == 0][feat]
        data_churn = df[df["Churn"] == 1][feat]
        bp = axes[1].boxplot(
            [data_stay, data_churn],
            patch_artist=True,
            medianprops=dict(color="#e8f0fe", linewidth=2),
            whiskerprops=dict(color="#3a5a7a"),
            capprops=dict(color="#3a5a7a"),
            flierprops=dict(marker=".", color="#3a5a7a", alpha=0.3, markersize=2),
        )
        for patch, color in zip(bp["boxes"], [STAY_C, CHURN_C]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes[1].set_xticks([1, 2])
        axes[1].set_xticklabels(["Retained", "Churned"])
        axes[1].set_title(f"Boxplot — {feat}")
        axes[1].spines[["top", "right"]].set_visible(False)
        for ax in axes:
            ax.set_facecolor("#0f1829")
            fig.patch.set_facecolor("#0f1829")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
# ══════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════
elif nav == "🤖  Model Performance":
    st.markdown("# Model Performance")
    st.markdown("<p style='color:#5a80a8;font-size:0.9rem;margin-top:-10px;margin-bottom:24px;'>Compare trained models side by side</p>", unsafe_allow_html=True)
 
    tab1, tab2, tab3 = st.tabs(["METRICS", "CONFUSION MATRIX", "ROC CURVE"])
 
    with tab1:
        st.markdown("### Performance Summary")
        rows = []
        for name, m in metrics.items():
            rows.append({
                "Model": name,
                "Accuracy": f"{m['Accuracy']}%",
                "Precision": f"{m['Precision']}%",
                "Recall": f"{m['Recall']}%",
                "F1 Score": f"{m['F1 Score']}%",
                "AUC-ROC": f"{m['AUC']}%",
            })
        st.dataframe(
            pd.DataFrame(rows).set_index("Model"),
            use_container_width=True,
        )
 
        st.markdown("### Metric Comparison Chart")
        metric_keys = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC"]
        model_names = list(metrics.keys())
        x = np.arange(len(metric_keys))
        width = 0.3
 
        fig, ax = plt.subplots(figsize=(10, 5))
        colors_bar = [ACCENT, "#f39c12"]
        for i, (name, color) in enumerate(zip(model_names, colors_bar)):
            vals = [metrics[name][k] for k in metric_keys]
            bars = ax.bar(x + i * width, vals, width,
                          label=name, color=color, alpha=0.85, edgecolor="none")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.3,
                        f"{v:.0f}", ha="center", va="bottom",
                        fontsize=7.5, color="#e8f0fe")
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(metric_keys)
        ax.set_ylim(0, 115)
        ax.set_ylabel("Score (%)")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="y", alpha=0.25)
        fig.patch.set_facecolor("#0f1829")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
    with tab2:
        st.markdown("### Confusion Matrices")
        col1, col2 = st.columns(2)
        for col, (name, m) in zip([col1, col2], metrics.items()):
            with col:
                st.markdown(f"**{name}**")
                cm = confusion_matrix(m["y_test"], m["y_pred"])
                fig, ax = plt.subplots(figsize=(4.5, 3.8))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                            linewidths=1, linecolor="#0a0e1a",
                            cbar=False, ax=ax,
                            annot_kws={"size": 14, "weight": "bold"})
                ax.set_xlabel("Predicted", labelpad=8)
                ax.set_ylabel("Actual", labelpad=8)
                ax.set_xticklabels(["Retained", "Churned"], rotation=0)
                ax.set_yticklabels(["Retained", "Churned"], rotation=0)
                fig.patch.set_facecolor("#0f1829")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
 
    with tab3:
        st.markdown("### ROC Curves")
        fig, ax = plt.subplots(figsize=(7, 5))
        colors_roc = [ACCENT, "#f39c12"]
        for (name, m), color in zip(metrics.items(), colors_roc):
            fpr, tpr, _ = roc_curve(m["y_test"], m["y_prob"])
            ax.plot(fpr, tpr, color=color, linewidth=2.2,
                    label=f"{name}  (AUC = {m['AUC']/100:.3f})")
        ax.plot([0, 1], [0, 1], color="#3a5a7a", linestyle="--",
                linewidth=1, label="Random Baseline")
        ax.fill_between([0, 1], [0, 1], alpha=0.04, color="#3a5a7a")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve Comparison")
        ax.legend(frameon=False, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=0.2)
        fig.patch.set_facecolor("#0f1829")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
 
# ══════════════════════════════════════════════
# PAGE: PREDICT CUSTOMER
# ══════════════════════════════════════════════
elif nav == "🔮  Predict Customer":
    st.markdown("# Customer Churn Predictor")
    st.markdown("<p style='color:#5a80a8;font-size:0.9rem;margin-top:-10px;margin-bottom:24px;'>Enter customer details to predict churn probability</p>", unsafe_allow_html=True)
 
    selected_model = st.selectbox("Select Model", list(models.keys()))
    st.markdown("<br>", unsafe_allow_html=True)
 
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 👤 Demographics")
        gender       = st.selectbox("Gender", ["Male", "Female"])
        senior       = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner      = st.selectbox("Has Partner", ["No", "Yes"])
        dependents   = st.selectbox("Has Dependents", ["No", "Yes"])
        tenure       = st.slider("Tenure (months)", 0, 72, 12)
 
    with col2:
        st.markdown("### 📶 Services")
        phone        = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines  = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet     = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        security     = st.selectbox("Online Security", ["No", "Yes"])
        backup       = st.selectbox("Online Backup", ["No", "Yes"])
        protection   = st.selectbox("Device Protection", ["No", "Yes"])
        support      = st.selectbox("Tech Support", ["No", "Yes"])
        tv           = st.selectbox("Streaming TV", ["No", "Yes"])
        movies       = st.selectbox("Streaming Movies", ["No", "Yes"])
 
    with col3:
        st.markdown("### 💳 Billing")
        contract     = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        paperless    = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment      = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"
        ])
        avg_charge   = st.slider("Avg Monthly Charge ($)", 18.0, 115.0, 65.0, 0.5)
 
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  PREDICT CHURN RISK")
 
    if predict_btn:
        # Build input dict
        yes_no = lambda v: 1 if v == "Yes" else 0
        multi_map  = {"No phone service": 0, "No": 1, "Yes": 2}
        inet_map   = {"No": 0, "DSL": 1, "Fiber optic": 2}
        cont_map   = {"Month-to-month": 0, "One year": 1, "Two year": 2}
 
        services_list = [phone, security, backup, protection, support, tv, movies]
        total_svc = sum(yes_no(s) for s in services_list)
        total_charges_est = avg_charge * (tenure + 1)
 
        input_data = {
            "gender":          1 if gender == "Male" else 0,
            "SeniorCitizen":   yes_no(senior),
            "Partner":         yes_no(partner),
            "Dependents":      yes_no(dependents),
            "tenure":          float(tenure),
            "PhoneService":    yes_no(phone),
            "MultipleLines":   multi_map[multi_lines],
            "InternetService": inet_map[internet],
            "OnlineSecurity":  yes_no(security),
            "OnlineBackup":    yes_no(backup),
            "DeviceProtection":yes_no(protection),
            "TechSupport":     yes_no(support),
            "StreamingTV":     yes_no(tv),
            "StreamingMovies": yes_no(movies),
            "Contract":        cont_map[contract],
            "PaperlessBilling":yes_no(paperless),
            "PaymentMethod_Bank transfer (automatic)": 1 if payment == "Bank transfer (automatic)" else 0,
            "PaymentMethod_Credit card (automatic)":   1 if payment == "Credit card (automatic)" else 0,
            "PaymentMethod_Electronic check":          1 if payment == "Electronic check" else 0,
            "PaymentMethod_Mailed check":              1 if payment == "Mailed check" else 0,
            "IsNewCustomer":       1 if tenure <= 6 else 0,
            "IsLongTermCustomer":  1 if tenure >= 48 else 0,
            "AvgMonthlyCharge":    float(avg_charge),
            "TotalServices":       int(total_svc),
        }
        input_df = pd.DataFrame([input_data])[feature_cols]
 
        mdl = models[selected_model]
        prob     = mdl.predict_proba(input_df)[0][1]
        pred     = int(prob >= 0.5)
        prob_pct = prob * 100
 
        st.markdown("<br>", unsafe_allow_html=True)
        res_col, explain_col = st.columns([1, 1.4])
 
        with res_col:
            if pred == 1:
                bar_style = "prob-bar-inner-churn"
                card_class = "pred-card-churn"
                emoji = "⚠️"
                verdict = "HIGH CHURN RISK"
                color = "#e74c3c"
                msg = "This customer is likely to leave. Consider a retention offer."
            else:
                bar_style = "prob-bar-inner-stay"
                card_class = "pred-card-stay"
                emoji = "✅"
                verdict = "LOW CHURN RISK"
                color = "#27ae60"
                msg = "This customer is likely to stay. No immediate action needed."
 
            st.markdown(f"""
            <div class="{card_class}">
                <div class="pred-title" style="color:{color};">{emoji} {verdict}</div>
                <div class="pred-sub">{msg}</div>
                <div style="margin-top:18px;">
                    <div style="display:flex;justify-content:space-between;font-size:0.78rem;color:#7a9cc6;margin-bottom:4px;">
                        <span>CHURN PROBABILITY</span><span style="color:{color};font-family:'Space Mono',monospace;font-weight:700;">{prob_pct:.1f}%</span>
                    </div>
                    <div class="prob-bar-outer">
                        <div class="{bar_style}" style="width:{prob_pct:.1f}%;"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
 
        with explain_col:
            st.markdown("### Risk Factor Summary")
            factors = []
            if cont_map[contract] == 0:
                factors.append(("📋 Month-to-month contract", "High Risk", CHURN_C))
            elif cont_map[contract] == 2:
                factors.append(("📋 Two-year contract", "Loyalty Signal", STAY_C))
            if tenure <= 6:
                factors.append(("⏱️ New customer (≤ 6 months)", "Elevated Risk", CHURN_C))
            elif tenure >= 48:
                factors.append(("⏱️ Long-term customer (≥ 48 mo)", "Loyalty Signal", STAY_C))
            if inet_map[internet] == 2:
                factors.append(("🌐 Fiber Optic subscriber", "Moderate Risk", "#f39c12"))
            if payment == "Electronic check":
                factors.append(("💳 Electronic check payment", "Slightly Higher Risk", "#f39c12"))
            if total_svc == 0:
                factors.append(("📦 No services subscribed", "High Risk", CHURN_C))
            elif total_svc >= 4:
                factors.append(("📦 Many services subscribed", "Engagement Signal", STAY_C))
 
            if not factors:
                factors.append(("ℹ️ No strong risk signals detected", "Neutral", ACCENT))
 
            for label, tag, color in factors:
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                     background:#0f1829;border:1px solid #1e2d4a;border-left:3px solid {color};
                     border-radius:6px;padding:10px 14px;margin-bottom:8px;">
                    <span style="font-size:0.85rem;">{label}</span>
                    <span style="font-size:0.72rem;color:{color};letter-spacing:0.06em;">{tag}</span>
                </div>
                """, unsafe_allow_html=True)