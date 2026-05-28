# ⚡ EnergyFlow - Power Consumption Prediction Using Machine Learning

## 📌 Project Overview

EnergyFlow is a Machine Learning project focused on predicting **Global Active Power Consumption** using household electricity consumption data.

The project includes:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Model Training
* Hyperparameter Tuning
* Model Comparison
* Model Evaluation
* Saving Trained Models

The primary goal of this project is to compare multiple machine learning regression models and identify the best-performing model for predicting household power consumption.

A future goal of this project is to build a web application for real-time electricity consumption prediction.

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
│
├── src/
│   ├── train_xgboost.py
│   ├── train_randomforest.py
│   └── train_linearregression.py
│
├── README.md
├── requirements.txt
└── .gitignore
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

# ⚙️ Data Preprocessing

### Missing Value Handling

Missing values were handled using:

* Time-based interpolation

This method was selected because the dataset represents sequential electricity consumption data where nearby timestamps are highly related.

### Feature Scaling

Feature scaling was applied only for linear models such as:

* Linear Regression
* Ridge Regression

Tree-based models like:

* Random Forest
* XGBoost

do not require feature scaling.

---

# 🤖 Machine Learning Models Implemented

The following regression models were implemented and evaluated:

* Linear Regression
* Ridge Regression
* Random Forest Regressor
* XGBoost Regressor

---

# 📈 Model Performance Comparison

| Model                   | Train RMSE | Test RMSE | Train R² Score | Test R² Score | Observation                                   |
| ----------------------- | ---------- | --------- | -------------- | ------------- | --------------------------------------------- |
| Linear Regression       | 0.3839     | 0.3859    | 0.7792         | 0.7704        | Stable performance but underfitting           |
| Ridge Regression        | 0.3839     | 0.3859    | 0.7792         | 0.7704        | Very similar performance to Linear Regression |
| Random Forest Regressor | 0.1148     | 0.3107    | 0.9802         | 0.8511        | Strong overfitting observed                   |
| XGBoost Regressor       | 0.2355     | 0.2742    | 0.9169         | 0.8840        | Best generalization performance               |

---

# 🏆 Best Performing Model

After comparing multiple regression models:

* XGBoost achieved the best balance between training and testing performance.
* Random Forest achieved extremely high training accuracy but showed stronger overfitting.
* Linear models generalized consistently but underperformed due to the non-linear nature of the dataset.

Final selected model:

```python
XGBRegressor(
    n_estimators=800,
    max_depth=3,
    learning_rate=0.1
)
```

### Final XGBoost Performance

#### Train Metrics

* RMSE: 0.2355
* R² Score: 0.9169

#### Test Metrics

* RMSE: 0.2742
* R² Score: 0.8840

---

# 🔍 Key Observations

### Linear Regression

* Showed minimal overfitting
* Lower predictive capability
* Indicates the dataset contains complex non-linear relationships

### Random Forest Regressor

* Achieved very high training performance
* Large train-test performance gap
* Indicates stronger overfitting on this dataset

### XGBoost Regressor

* Achieved the best generalization
* Reduced overfitting compared to Random Forest
* Performed best overall on unseen data

---

# 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Pickle

---

# 🚀 Future Improvements

Planned future improvements include:

* Implementing additional ML models
* Time-series forecasting workflows
* Chronological train-test splitting
* Lag feature engineering
* Advanced hyperparameter optimization
* Feature importance analysis
* Pipeline implementation
* Building a web application for predictions
* Deployment using Flask or FastAPI

---

# 📚 Key Learnings

This project helped in understanding:

* Data preprocessing workflows
* Feature engineering
* Exploratory data analysis
* Regression modeling
* Feature scaling concepts
* Hyperparameter tuning
* Model comparison and benchmarking
* Overfitting vs underfitting
* Generalization performance
* Machine learning project structure
* Git and GitHub workflow for ML projects

---

# 📌 Conclusion

This project demonstrates a complete end-to-end machine learning workflow for regression-based power consumption prediction.

After comparing multiple regression models, XGBoost emerged as the best-performing model due to its superior generalization capability and balanced performance on unseen data.

The project also provided valuable practical understanding of:

* Overfitting
* Underfitting
* Model complexity
* Hyperparameter tuning
* Regression model comparison

Further improvements will focus on time-series forecasting techniques and deployment of the trained model as a web application.
