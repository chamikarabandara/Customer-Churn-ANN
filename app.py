
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="Customer Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    PROJECT_DIR / "data" / "processed" / "telco_churn_cleaned.csv"
)

# ==================================================
# CUSTOM CSS
# ==================================================
st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
    color: #1E293B;
}

[data-testid="stHeader"] {
    background-color: #F4F7FB;
}

[data-testid="stSidebar"] {
    background-color: #14213D;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}

.stApp h1, .stApp h2, .stApp h3,
.stApp p, .stApp label {
    color: #1E293B;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #E2E8F0;
    box-shadow: 0 3px 10px rgba(15, 23, 42, 0.05);
}

div[data-testid="stMetric"] label {
    color: #64748B !important;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #14213D !important;
    font-weight: 700;
}

div[data-testid="stMetricDelta"] {
    color: #64748B !important;
}

div[data-testid="stPlotlyChart"] {
    background-color: #FFFFFF;
    border-radius: 14px;
    padding: 10px;
    border: 1px solid #E2E8F0;
}

hr {
    border-color: #DCE3ED;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# CHART STYLE
# ==================================================
def style_chart(fig, height=400):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="Arial, sans-serif",
            color="#1E293B",
            size=13
        ),
        title_font=dict(
            size=18,
            color="#14213D"
        ),
        legend=dict(
            font=dict(
                color="#1E293B",
                size=12
            ),
            bgcolor="rgba(255,255,255,0.9)"
        ),
        height=height,
        margin=dict(t=35, b=45, l=35, r=25),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            font_color="#1E293B"
        )
    )

    fig.update_xaxes(
        showgrid=False,
        linecolor="#CBD5E1",
        tickfont=dict(color="#475569"),
        title_font=dict(color="#1E293B")
    )

    fig.update_yaxes(
        gridcolor="#E2E8F0",
        zerolinecolor="#CBD5E1",
        tickfont=dict(color="#475569"),
        title_font=dict(color="#1E293B")
    )

    return fig

# ==================================================
# SIDEBAR
# ==================================================
with st.sidebar:
    st.title("📊 Churn Analytics")
    st.caption("Customer Intelligence Platform")
    st.divider()

    st.markdown("### Navigation")
    st.markdown("🏠 Overview")
    st.markdown("🔮 Predict Churn")
    st.markdown("📈 Model Insights")
    st.markdown("🔍 Data Explorer")

    st.divider()
    st.caption("Machine Learning Project")
    st.caption("Telco Customer Churn")

# ==================================================
# LOAD DATA
# ==================================================
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

st.title("Customer Churn Dashboard")
st.markdown(
    "Monitor customer retention, explore churn patterns, "
    "and understand customer behaviour."
)

if not DATA_PATH.exists():
    st.error(f"Dataset not found: {DATA_PATH}")
    st.info(
        "Please check that "
        "data/processed/telco_churn_cleaned.csv exists."
    )
    st.stop()

df = load_data(str(DATA_PATH))

if "Churn" not in df.columns:
    st.error("The dataset must contain a 'Churn' column.")
    st.stop()

# ==================================================
# PREPARE CHURN LABELS
# ==================================================
if pd.api.types.is_numeric_dtype(df["Churn"]):
    churn_flag = df["Churn"].eq(1)
else:
    churn_flag = (
        df["Churn"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "1", "true", "churn"])
    )

total_customers = len(df)
churned_customers = int(churn_flag.sum())
retained_customers = total_customers - churned_customers

churn_rate = (
    churned_customers / total_customers * 100
    if total_customers > 0 else 0
)
retention_rate = 100 - churn_rate

# ==================================================
# KPI CARDS
# ==================================================
st.subheader("Key Performance Indicators")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Churned Customers", f"{churned_customers:,}")
c3.metric("Churn Rate", f"{churn_rate:.1f}%")
c4.metric("Retention Rate", f"{retention_rate:.1f}%")

st.divider()

# ==================================================
# CUSTOMER CHURN DISTRIBUTION
# ==================================================
left, right = st.columns(2)

with left:
    st.subheader("Customer Churn Distribution")

    distribution = pd.DataFrame({
        "Customer Status": ["Retained", "Churned"],
        "Customers": [
            retained_customers,
            churned_customers
        ]
    })

    fig = px.pie(
        distribution,
        names="Customer Status",
        values="Customers",
        hole=0.62,
        color="Customer Status",
        color_discrete_map={
            "Retained": "#2A9D8F",
            "Churned": "#E76F51"
        }
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent",
        textfont=dict(color="#FFFFFF", size=13),
        hovertemplate=(
            "%{label}<br>"
            "Customers: %{value:,}<br>"
            "Percentage: %{percent}"
            "<extra></extra>"
        )
    )

    fig = style_chart(fig, height=380)
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# CHURN BY CONTRACT
# ==================================================
with right:
    st.subheader("Churn by Contract Type")

    if "Contract" in df.columns:
        contract_df = df.copy()
        contract_df["_ChurnFlag"] = churn_flag.astype(int)

        contract_summary = (
            contract_df.groupby("Contract", dropna=False)
            .agg(
                Customers=("_ChurnFlag", "size"),
                Churn_Rate=("_ChurnFlag", "mean")
            )
            .reset_index()
        )

        contract_summary["Churn Rate (%)"] = (
            contract_summary["Churn_Rate"] * 100
        )

        fig = px.bar(
            contract_summary,
            x="Contract",
            y="Churn Rate (%)",
            text=contract_summary["Churn Rate (%)"].round(1),
            color="Contract",
            color_discrete_sequence=[
                "#1D4ED8",
                "#2A9D8F",
                "#E9A23B"
            ],
            hover_data={"Customers": True}
        )

        fig.update_traces(
            texttemplate="%{text}%",
            textposition="outside",
            cliponaxis=False,
            textfont=dict(color="#1E293B")
        )

        fig.update_layout(
            showlegend=False,
            yaxis_title="Churn Rate (%)",
            xaxis_title="Contract Type"
        )

        fig = style_chart(fig, height=380)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Contract column is not available.")

# ==================================================
# MONTHLY CHARGES ANALYSIS
# ==================================================
st.subheader("Monthly Charges by Customer Status")

if "MonthlyCharges" in df.columns:
    charge_df = df.copy()

    charge_df["_CustomerStatus"] = churn_flag.map({
        True: "Churned",
        False: "Retained"
    })

    fig = px.histogram(
        charge_df,
        x="MonthlyCharges",
        color="_CustomerStatus",
        barmode="overlay",
        opacity=0.72,
        nbins=35,
        color_discrete_map={
            "Retained": "#2A9D8F",
            "Churned": "#E76F51"
        },
        labels={
            "MonthlyCharges": "Monthly Charges ($)",
            "_CustomerStatus": "Customer Status"
        }
    )

    fig.update_layout(
        xaxis_title="Monthly Charges ($)",
        yaxis_title="Number of Customers",
        bargap=0.05
    )

    fig = style_chart(fig, height=430)
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# FOOTER
# ==================================================
st.divider()

st.caption(
    "Customer Churn Analytics | "
    "Telco Customer Churn Dataset | "
    "Developed with Python and Streamlit"
)
