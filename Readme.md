# ⚡ EnergyFlow - Power Consumption Prediction Using Machine Learning

## 📌 Project Overview

EnergyFlow is a Machine Learning project focused on predicting **Global Active Power Consumption** using household electricity consumption data.

The project includes:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Training using XGBoost
* Model Evaluation
* Saving Trained Models

The long-term goal of this project is to compare multiple machine learning models and build a web application for real-time power consumption prediction.

---

# 📂 Project Structure

```bash
EnergyFlow-PML_Project/
│
├── data/
│   ├── Raw_Electricity_Data.csv
│   └── cleaned_data.csv
│
├── notebooks/
│   └── DataProcessing&EDA.ipynb
│
├── models/
│   └── XGBmodel.pkl
│
├── src/
│   └── train_xgboost.py
│
├── README.md
└── requirements.txt
```

---

# 📊 Exploratory Data Analysis (EDA)

The EDA process included:

* Handling missing values using time-based interpolation
* Datetime feature extraction
* Correlation analysis
* Outlier analysis
* Multicollinearity analysis using VIF
* Trend analysis across years, months, and days

### Feature Engineering

Extracted temporal features such as:

* Year
* Month
* Day
* Hour
* Minute

The original `Date` and `Time` columns were removed after extracting relevant numerical features.

---

# 🤖 Machine Learning Model

Currently Implemented:

* XGBoost Regressor

### Hyperparameters Used

```python
XGBRegressor(
    n_estimators=800,
    max_depth=3,
    learning_rate=0.1
)
```

---

# 📈 Model Performance

## Train Metrics

* RMSE: 0.2355
* R² Score: 0.9169

## Test Metrics

* RMSE: 0.2742
* R² Score: 0.8840

The model demonstrates good generalization performance with reduced overfitting.

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost

---

# 🚀 Future Improvements

Planned future improvements include:

* Implementing additional ML models
* Model comparison and benchmarking
* Time-series forecasting techniques
* Lag feature engineering
* Hyperparameter optimization
* Building a web application for predictions
* Deployment

---

# 📌 Key Learnings

This project helped in understanding:

* Data preprocessing workflows
* Feature engineering
* Model evaluation
* Overfitting analysis
* Machine learning project structure
* Regression modeling using XGBoost

---
