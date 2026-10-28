
import streamlit as st
import pandas as pd
from pathlib import Path

st.set_page_config(
    page_title="Data Explorer",
    page_icon="🔎",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "processed" / "telco_churn_cleaned.csv"

# ---------------- STYLING ----------------
st.markdown("""
<style>
    .stApp {
        background-color: #F4F7FB;
        color: #172B4D;
    }

    [data-testid="stSidebar"] {
        background-color: #101B35;
    }

    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    h1, h2, h3 {
        color: #142B52 !important;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px;
        min-height: 105px;
        box-shadow: 0 3px 10px rgba(15, 23, 42, 0.04);
    }

    .metric-label {
        color: #64748B;
        font-size: 14px;
    }

    .metric-value {
        color: #173B70;
        font-size: 27px;
        font-weight: 700;
        margin-top: 8px;
    }

    /* Filter labels */
    [data-testid="stWidgetLabel"] p {
        color: #142B52 !important;
        font-weight: 600 !important;
    }

    /* Search box */
    .stTextInput input {
        color: #142B52 !important;
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
    }

    .stTextInput input::placeholder {
        color: #64748B !important;
        opacity: 1;
    }

    /* Dropdown boxes */
    div[data-testid="stSelectbox"] div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
    }

    div[data-testid="stSelectbox"] div[data-baseweb="select"] * {
        color: #142B52 !important;
    }

    /* Dropdown menu options */
    div[data-baseweb="popover"] ul,
    div[data-baseweb="popover"] li {
        background-color: #FFFFFF !important;
        color: #142B52 !important;
    }

    /* Download button */
    .stDownloadButton button {
        background-color: #173B70 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    .stDownloadButton button p {
        color: #FFFFFF !important;
    }

    .stDownloadButton button:hover {
        background-color: #2455A4 !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🔎 Data Explorer")
st.write("Explore, search, filter and download customer churn data.")

# ---------------- LOAD DATA ----------------
if not DATA_PATH.exists():
    st.error(
        f"Dataset not found at: {DATA_PATH}\n\n"
        "Check that telco_churn_cleaned.csv exists "
        "inside data/processed/."
    )
    st.stop()

df = pd.read_csv(DATA_PATH)

if "Churn" not in df.columns:
    st.error("The dataset must contain a 'Churn' column.")
    st.stop()

# ---------------- NORMALIZE CHURN ----------------
# Preserve original values for display, but normalize
# values internally for accurate counting and filtering.
raw_churn = df["Churn"].astype("string").str.strip().str.lower()

churn_mapping = {
    "yes": "Yes",
    "no": "No",
    "1": "Yes",
    "0": "No",
    "true": "Yes",
    "false": "No"
}

normalized_churn = raw_churn.map(churn_mapping)

# Use normalized labels only if every non-missing value is recognized.
recognized = raw_churn.isna() | normalized_churn.notna()

if recognized.all():
    df["Churn"] = normalized_churn

# ---------------- DATASET OVERVIEW ----------------
st.subheader("Dataset Overview")

total_customers = len(df)
total_columns = len(df.columns)

normalized_for_count = (
    df["Churn"].astype("string").str.strip().str.lower()
)

churned_customers = normalized_for_count.isin(
    ["yes", "1", "true"]
).sum()

churn_rate = (
    churned_customers / total_customers * 100
    if total_customers > 0 else 0
)

m1, m2, m3, m4 = st.columns(4)

cards = [
    (m1, "Total Records", f"{total_customers:,}"),
    (m2, "Total Columns", f"{total_columns:,}"),
    (m3, "Churned Customers", f"{churned_customers:,}"),
    (m4, "Churn Rate", f"{churn_rate:.2f}%")
]

for col, label, value in cards:
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# ---------------- FILTERS ----------------
st.subheader("Search and Filter Customers")

filter_col1, filter_col2, filter_col3 = st.columns(3)

with filter_col1:
    search_text = st.text_input(
        "Search Customer ID",
        placeholder="Enter customer ID..."
    )

with filter_col2:
    churn_options = ["All"] + sorted(
        df["Churn"].dropna().astype(str).unique().tolist()
    )

    selected_churn = st.selectbox(
        "Churn Status",
        churn_options
    )

with filter_col3:
    if "Contract" in df.columns:
        contract_options = ["All"] + sorted(
            df["Contract"].dropna().astype(str).unique().tolist()
        )

        selected_contract = st.selectbox(
            "Contract Type",
            contract_options
        )
    else:
        selected_contract = "All"
        st.info("Contract column is unavailable.")

# ---------------- APPLY FILTERS ----------------
filtered_df = df.copy()

if search_text:
    if "customerID" in filtered_df.columns:
        filtered_df = filtered_df[
            filtered_df["customerID"].astype(str).str.contains(
                search_text,
                case=False,
                na=False,
                regex=False
            )
        ]
    else:
        st.warning("Customer ID column is unavailable.")

if selected_churn != "All":
    filtered_df = filtered_df[
        filtered_df["Churn"].astype(str) == selected_churn
    ]

if selected_contract != "All" and "Contract" in filtered_df.columns:
    filtered_df = filtered_df[
        filtered_df["Contract"].astype(str) == selected_contract
    ]

# ---------------- DATA TABLE ----------------
st.subheader("Customer Records")
st.write(f"Showing **{len(filtered_df):,}** matching records.")

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True,
    height=400
)

# ---------------- DOWNLOAD ----------------
csv_data = filtered_df.to_csv(index=False).encode("utf-8-sig")

st.download_button(
    label="📥 Download Filtered Data (CSV)",
    data=csv_data,
    file_name="filtered_customer_churn_data.csv",
    mime="text/csv",
    use_container_width=False
)

# ---------------- COLUMN INFORMATION ----------------
with st.expander("View Column Information"):
    column_info = pd.DataFrame({
        "Column": df.columns,
        "Data Type": [str(dtype) for dtype in df.dtypes],
        "Missing Values": df.isna().sum().values
    })

    st.dataframe(
        column_info,
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "Customer data is displayed from the processed dataset. "
    "Check the source label encoding before interpreting churn statistics."
)
