# ⚡ EnergyFlow - Household Power Consumption Prediction

## 📖 Project Overview

EnergyFlow is a machine learning and time-series forecasting project focused on predicting household electricity consumption using historical power usage, weather conditions, and temporal features.

The project explores both traditional statistical forecasting techniques and modern machine learning algorithms to identify the most effective approach for modeling energy consumption patterns.

---

## 🎯 Objectives

* Analyze household electricity consumption behavior.
* Build predictive models for power consumption forecasting.
* Compare traditional time-series methods with machine learning approaches.
* Evaluate model performance using standard regression metrics.
* Identify the most suitable model for accurate energy consumption prediction.

---

## 📊 Dataset Features

The dataset contains electrical measurements, weather information, and time-based features, including:

### Electrical Features

* Global Reactive Power
* Voltage
* Sub Metering 1
* Sub Metering 2
* Sub Metering 3

### Weather Features

* Temperature
* Cloud Cover
* Wind
* Precipitation
* Snowfall

### Time Features

* Hour
* Day
* Month
* Year

### Target Variable

* Global Active Power

---

## 🤖 Models Implemented

### Machine Learning Models

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* XGBoost Regressor

### Time Series Models

* SARIMA
* SARIMAX (with exogenous variables)

---

## 📈 Model Performance

| Model                   | RMSE   | R² Score | Observation                                           |
| ----------------------- | ------ | -------- | ----------------------------------------------------- |
| Linear Regression       | 0.3859 | 0.7704   | Stable baseline performance                           |
| Ridge Regression        | 0.3859 | 0.7704   | Similar performance to Linear Regression              |
| Random Forest Regressor | 0.3107 | 0.8511   | Strong predictive performance with slight overfitting |
| XGBoost Regressor       | 0.2742 | 0.8840   | Best overall performance                              |
| SARIMA                  | 0.8979 | -0.0147  | Poor forecasting performance                          |
| SARIMAX                 | 0.3122 | 0.7797   | Significant improvement using external features       |

---

## ⏳ Time Series Analysis

### SARIMA

A SARIMA model was implemented using only historical values of the target variable.

Despite extensive parameter tuning and Auto-ARIMA optimization, the model struggled to capture the complex power consumption patterns present in the dataset.

Performance:

* RMSE: 0.8979
* R² Score: -0.0147

The negative R² score indicates that the model performed worse than a simple mean-based prediction.

### SARIMAX

To incorporate additional explanatory information, a SARIMAX model was trained using exogenous variables such as electrical measurements, weather data, and time-based features.

Performance improved substantially:

* RMSE: 0.3122
* MSE: 0.0974
* R² Score: 0.7797

These results demonstrate that household power consumption is strongly influenced by external variables and cannot be accurately modeled using past target values alone.

---

## 🔍 Key Findings

* XGBoost achieved the best predictive performance among all evaluated models.
* Traditional SARIMA forecasting was unable to model the dataset effectively.
* SARIMAX significantly improved forecasting accuracy by incorporating exogenous features.
* Weather conditions, electrical measurements, and temporal information contribute substantial predictive power.
* Machine learning models outperformed classical statistical forecasting methods for this dataset.

---

## 🛠 Technologies Used


* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Statsmodels
* Pmdarima
* Matplotlib
* Streamlit

---

## 📂 Project Structure

EnergyFlow/
│
├── Data/
│   ├── Raw_Electricity_Data.csv
│   └── cleaned_data.csv
│
├── Notebooks/
│   └── DataProcessing&EDA.ipynb
│
├── Src/
│   ├── linearRegression_training.py
│   ├── randomforest_training.py
│   ├── sarima_training.py
│   ├── sarimax_training.py
│   └── xgboost_training.py
│
├── models/
│   └── (excluded from repository)
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

> Note: Trained model files are excluded from version control and are therefore not included in this repository.

---

## 🚀 Future Improvements

- Deploy the Streamlit application to Streamlit Cloud.
- Integrate real-time weather APIs for live predictions.
- Add prediction history and visualization dashboards.
- Implement feature importance analysis using SHAP values.
- Explore advanced forecasting models such as Prophet.
- Experiment with deep learning architectures (LSTM, GRU, Transformer).
- Build multi-step energy consumption forecasting pipelines.
- Develop a REST API version using FastAPI or Flask.

---
## ▶️ Running the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```
if this commad is not working try :

```bash
python -m streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

## 📌 Conclusion

This project demonstrates both machine learning and classical time-series approaches for household power consumption prediction.

Among all evaluated models, XGBoost delivered the best overall performance with an R² Score of 0.8840 and the lowest prediction error.

While SARIMA struggled to model the complex consumption behavior, SARIMAX showed that incorporating external variables substantially improves forecasting performance. This highlights the importance of feature-rich modeling approaches when predicting household energy consumption.

Based on the evaluation results, XGBoost was selected as the final model due to its superior predictive accuracy and generalization capability.
