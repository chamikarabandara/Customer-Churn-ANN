# 📊 Customer Churn Prediction Using Artificial Neural Networks

An interactive Machine Learning dashboard for analyzing customer churn, predicting customer retention risks, comparing classification models, and understanding the factors influencing churn predictions.

## 🚀 Project Overview

Customer churn occurs when customers stop using a company's products or services. Identifying customers who are likely to leave helps businesses develop effective customer retention strategies.

This project uses the Telco Customer Churn dataset to explore customer behavior and build a predictive model using an Artificial Neural Network (ANN). The project also compares multiple machine learning algorithms and provides an interactive Streamlit dashboard for exploring data and generating predictions.

## 🎯 Project Objectives

- Analyze customer churn patterns and customer characteristics.
- Perform data cleaning and preprocessing.
- Develop an Artificial Neural Network model for churn prediction.
- Compare ANN performance with other machine learning models.
- Explain model predictions using SHAP.
- Build an interactive dashboard for customer analysis and prediction.
- Provide business recommendations to support customer retention.

## ✨ Dashboard Features

### 1. Overview Dashboard
- Total customer statistics.
- Customer churn and retention rates.
- Churn distribution visualizations.
- Churn analysis by contract type.
- Monthly charge comparisons.

### 2. Predict Churn
- Interactive customer information form.
- Customer churn probability prediction.
- Predicted churn or non-churn classification.
- Customer summary and prediction results.

### 3. Model Insights
- Comparison of Logistic Regression, ANN, Random Forest, and Support Vector Machine.
- Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.
- ANN confusion matrix visualization.
- SHAP-based model explainability.
- Business recommendations for customer retention.

### 4. Data Explorer
- Dataset overview and summary statistics.
- Search customers by customer ID.
- Filter records by churn status and contract type.
- Interactive customer data table.
- Download filtered customer records as a CSV file.
- View column data types and missing values.

## 🛠️ Technologies Used

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Model Explainability | SHAP |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib, Seaborn |
| Model Persistence | Joblib |
| Development Environment | Jupyter Notebook, VS Code |
| Version Control | Git and GitHub |

## 📂 Project Structure

```text
Customer_Churn_ANN/
│
├── app.py
│
├── pages/
│   ├── 2_Predict_Churn.py
│   ├── 3_Model_Insights.py
│   └── 4_Data_Explorer.py
│
├── data/
│   └── processed/
│       ├── telco_churn_cleaned.csv
│       ├── X_train.npy
│       ├── X_test.npy
│       ├── y_train.npy
│       └── y_test.npy
│
├── models/
│   ├── best_ann_model.keras
│   └── preprocessor.joblib
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_ann_model.ipynb
│   └── 05_shap_explainability.ipynb
│
├── reports/
│   ├── figures/
│   └── metrics/
│
├── requirements.txt
└── README.md
```

*Note: The structure shows the main project files. Optional files and generated reports may vary depending on the current repository contents.*

## 📊 Dataset Description

The project uses the Telco Customer Churn dataset.

- **Total records:** 7,043
- **Total columns:** 21
- **Target variable:** `Churn`
- **Customer identifier:** `customerID`

The dataset contains customer demographics, service subscriptions, contract information, billing details, tenure, and churn status.

### Important Features

| Feature | Description |
|---|---|
| `gender` | Customer gender |
| `SeniorCitizen` | Indicates whether the customer is a senior citizen |
| `Partner` | Whether the customer has a partner |
| `Dependents` | Whether the customer has dependents |
| `tenure` | Number of months the customer has stayed |
| `InternetService` | Type of internet service |
| `Contract` | Customer contract type |
| `MonthlyCharges` | Monthly amount charged |
| `TotalCharges` | Total amount charged |
| `Churn` | Customer churn status |

## ⚙️ Data Preprocessing

The preprocessing pipeline includes:

1. Removing the `customerID` column from model features.
2. Converting the churn target into binary values: Yes = 1 and No = 0.
3. Separating numerical and categorical features.
4. Applying `StandardScaler` to numerical features.
5. Applying `OneHotEncoder` to categorical features.
6. Splitting the dataset into training and testing sets.
7. Saving the trained preprocessor for reuse during prediction.

The transformed feature matrix contains 45 features.

## 🤖 Machine Learning Models

Four classification algorithms were evaluated:

- Logistic Regression
- Artificial Neural Network (ANN)
- Random Forest
- Support Vector Machine (SVM)

### Model Evaluation Results

The following results are from the recorded initial model comparison.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% | 84.19% |
| Artificial Neural Network | 79.13% | 61.36% | 57.75% | 59.50% | 83.83% |
| Random Forest | 76.30% | 54.37% | 66.58% | 59.86% | 82.46% |
| Support Vector Machine | 74.38% | 51.15% | 77.27% | 61.55% | 82.11% |

**Note:** These values represent the recorded initial comparison results. They may differ from later tuned model experiments.

## 📈 Model Explainability with SHAP

SHAP (SHapley Additive exPlanations) is used to investigate how input features influence model predictions.

The recorded SHAP summary highlights features such as:

- Customer tenure
- Monthly charges
- Contract type
- Internet service
- Technical support
- Total charges

SHAP visualizations help interpret model behavior. Feature influence does not necessarily establish causation.

## 💡 Business Recommendations

Based on customer churn analysis, businesses can consider:

- Identifying customers with high predicted churn risk.
- Reviewing month-to-month contract retention patterns.
- Investigating customer service and technical support issues.
- Evaluating monthly charges and customer value.
- Developing targeted customer retention campaigns.
- Monitoring churn patterns to support business decisions.

These recommendations should be validated against business outcomes before implementation.

## 💻 Installation and Setup

### Prerequisites

- Python 3.11 recommended for the current project environment.
- Git
- Visual Studio Code or another Python development environment.

### 1. Clone the Repository

```bash
git clone https://github.com/chamikarabandara/Customer-Churn-ANN.git
```

### 2. Navigate to the Project Directory

```bash
cd Customer-Churn-ANN
```

### 3. Create a Virtual Environment

**Windows:**

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard should open in your browser at the local URL displayed in the terminal.

## 📁 Required Files

The application expects the following files:

- `data/processed/telco_churn_cleaned.csv`
- `models/best_ann_model.keras`
- `models/preprocessor.joblib`
- `reports/figures/ann_confusion_matrix.png`
- `reports/figures/ann_shap_summary.png`

Ensure these files are available at the expected paths. If a required model or dataset is not included in the repository, add it or update the application to load it from an appropriate location.

## 🔮 Future Improvements

- Deploy the dashboard using Streamlit Community Cloud.
- Improve ANN performance through additional hyperparameter tuning.
- Add more interactive customer churn visualizations.
- Introduce additional customer segmentation analysis.
- Evaluate prediction thresholds to balance precision and recall.
- Add automated model monitoring and performance tracking.

## 👩‍💻 Author

**Chamikara Bandara**

GitHub: [@chamikarabandara](https://github.com/chamikarabandara)

## 📌 Repository

[Customer Churn Prediction Using ANN — GitHub](https://github.com/chamikarabandara/Customer-Churn-ANN)

---

*This project was developed for educational and portfolio purposes to demonstrate machine learning, deep learning, model explainability, data analysis, and interactive dashboard development.*
