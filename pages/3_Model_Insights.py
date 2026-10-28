
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(
    page_title="Model Insights",
    page_icon="📊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parents[1]

# ---------------- PAGE STYLING ----------------
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

    .page-subtitle {
        color: #64748B;
        font-size: 16px;
        margin-bottom: 24px;
    }

    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px 18px;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.05);
        min-height: 125px;
    }

    .metric-label {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #173B70;
        font-size: 28px;
        font-weight: 700;
    }

    .section-note {
        color: #64748B;
        font-size: 14px;
    }

    div[data-testid="stDataFrame"] {
        background: #FFFFFF;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("📊 Model Insights")

st.markdown(
    '<p class="page-subtitle">'
    'Evaluate machine learning performance and understand customer churn predictions.'
    '</p>',
    unsafe_allow_html=True
)

# ---------------- MODEL RESULTS ----------------
model_results = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Artificial Neural Network",
        "Random Forest",
        "Support Vector Machine"
    ],
    "Accuracy": [0.8055, 0.7913, 0.7630, 0.7438],
    "Precision": [0.6572, 0.6136, 0.5437, 0.5115],
    "Recall": [0.5588, 0.5775, 0.6658, 0.7727],
    "F1-Score": [0.6040, 0.5950, 0.5986, 0.6155],
    "ROC-AUC": [0.8419, 0.8383, 0.8246, 0.8211]
})

# ---------------- ANN METRIC CARDS ----------------
st.subheader("Artificial Neural Network Performance")
st.caption("Performance metrics from the current ANN evaluation.")

metrics = [
    ("Accuracy", 0.7913),
    ("Precision", 0.6136),
    ("Recall", 0.5775),
    ("F1-Score", 0.5950),
    ("ROC-AUC", 0.8383)
]

cols = st.columns(5)

for col, (label, value) in zip(cols, metrics):
    col.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value:.2%}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.write("")

# ---------------- MODEL COMPARISON ----------------
st.subheader("Machine Learning Model Comparison")

st.markdown(
    '<p class="section-note">'
    'Compare the four models across different evaluation metrics.'
    '</p>',
    unsafe_allow_html=True
)

display_df = model_results.copy()

metric_columns = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC"
]

for column in metric_columns:
    display_df[column] = display_df[column].map(
        lambda value: f"{value:.2%}"
    )

st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

selected_metric = st.selectbox(
    "Choose an evaluation metric",
    metric_columns,
    index=4
)

chart_df = model_results.sort_values(
    selected_metric,
    ascending=True
)

# ---------------- PROFESSIONAL BAR CHART ----------------
fig = px.bar(
    chart_df,
    x=selected_metric,
    y="Model",
    orientation="h",
    text=selected_metric,
    color=selected_metric,
    color_continuous_scale=["#B8D4F0", "#2455A4"],
    template="plotly_white",
    title=f"Model Comparison — {selected_metric}"
)

fig.update_traces(
    texttemplate="%{text:.2%}",
    textposition="outside",
    cliponaxis=False,
    textfont=dict(
        color="#000000",
        size=12
    )
)

fig.update_layout(
    height=430,

    margin=dict(
        l=20,
        r=80,
        t=70,
        b=45
    ),

    # X-axis: black labels
    xaxis=dict(
        range=[0, 1],
        tickformat=".0%",
        title=selected_metric,
        title_font=dict(
            color="#000000",
            size=14
        ),
        tickfont=dict(
            color="#000000",
            size=12
        ),
        gridcolor="#E2E8F0",
        zerolinecolor="#64748B",
        showline=True,
        linecolor="#CBD5E1"
    ),

    # Y-axis: model names in black
    yaxis=dict(
        title="",
        tickfont=dict(
            color="#000000",
            size=13
        ),
        showline=False,
        automargin=True
    ),

    # White chart background
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",

    # All chart text in black
    font=dict(
        family="Arial, sans-serif",
        color="#000000",
        size=13
    ),

    title_font=dict(
        color="#000000",
        size=18
    ),

    coloraxis_showscale=False,
    showlegend=False
)

st.plotly_chart(
    fig,
    use_container_width=True,
    config={"displaylogo": False}
)

# ---------------- CONFUSION MATRIX ----------------
st.subheader("ANN Confusion Matrix")

st.markdown(
    '<p class="section-note">'
    'Review the ANN model’s correct and incorrect classifications.'
    '</p>',
    unsafe_allow_html=True
)

confusion_path = (
    BASE_DIR / "reports" / "figures" / "ann_confusion_matrix.png"
)

if confusion_path.exists():
    with st.container(border=True):
        st.image(
            str(confusion_path),
            caption="ANN model confusion matrix",
            use_container_width=True
        )
else:
    st.warning(
        "Confusion matrix image not found. "
        "Check reports/figures/ann_confusion_matrix.png."
    )

# ---------------- SHAP EXPLAINABILITY ----------------
st.subheader("SHAP Model Explainability")

st.markdown(
    '<p class="section-note">'
    'Explore which features influence the ANN model predictions.'
    '</p>',
    unsafe_allow_html=True
)

shap_path = (
    BASE_DIR / "reports" / "figures" / "ann_shap_summary.png"
)

if shap_path.exists():
    with st.container(border=True):
        st.image(
            str(shap_path),
            caption="SHAP summary plot for the ANN model",
            use_container_width=True
        )
else:
    st.warning(
        "SHAP summary image not found. "
        "Check reports/figures/ann_shap_summary.png."
    )

# ---------------- BUSINESS RECOMMENDATIONS ----------------
st.subheader("Business Recommendations")

recommendations = [
    (
        "Identify at-risk customers",
        "Use churn probabilities to prioritize customers for retention campaigns."
    ),
    (
        "Review customer contracts",
        "Examine contract types and customer retention patterns."
    ),
    (
        "Evaluate service costs",
        "Investigate whether monthly charges are associated with churn risk."
    ),
    (
        "Improve customer support",
        "Review support service issues and identify opportunities to improve retention."
    )
]

# Display recommendations in two columns
for start in range(0, len(recommendations), 2):
    left, right = st.columns(2)

    for col, item in zip(
        [left, right],
        recommendations[start:start + 2]
    ):
        title, description = item

        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.write(description)

st.caption(
    "Note: Model metrics reflect the recorded evaluation results. "
    "Predictions support decision-making and should be validated before business use."
)
