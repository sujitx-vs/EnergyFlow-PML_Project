import pandas as pd
# from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import r2_score,root_mean_squared_error,mean_squared_error
import pickle as pk


data = pd.read_csv('Data/cleaned_data.csv')

X = data.drop('Global_active_power',axis=1)
Y = data['Global_active_power']

# X.drop('minutes',axis=1,inplace=True)
split_index = int(X.shape[0]*.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]


split_index = int(Y.shape[0]*.8)

Y_train = Y.iloc[:split_index]
Y_test = Y.iloc[split_index:]

# scaler = StandardScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
# scaling destroy the actual value importance of the data,so dont use it, tried it and the performance was poor.
# Scaling was tested but resulted in significantly lower
# forecasting performance for this dataset, so the original
# feature values were retained.

# corr = X_train.corr()
# print(corr)



model = SARIMAX(
    exog=X_train,
    endog=Y_train,
    order=(2,0,0),
    seasonal_order=(0,0,0,0)
)

# results = model.fit()

results = model.fit(
    maxiter=1000,
    method='powell'
)


# print(results.mle_retvals)

forecast = results.get_forecast(
    steps=len(Y_test),
    exog=X_test
)

pred = forecast.predicted_mean


rmse = root_mean_squared_error(Y_test, pred)
mse = mean_squared_error(Y_test, pred)
r2 = r2_score(Y_test, pred)

print("RMSE:", rmse)
print("MSE :", mse)
print("R²  :", r2)

# Performance
# RMSE: 0.3121569559865065
# MSE : 0.09744196517076174
# R²  : 0.7796535316062327

# performance increased well with respective of sarima model.
# basically it means the data also consider the features than a simple forecasting from previous target column.

# SARIMAX significantly outperformed SARIMA.
# This indicates that the exogenous variables contain
# substantial predictive information and that forecasting
# based only on past target values is insufficient for
# this dataset.

print("Converged:", results.mle_retvals["converged"])

with open("models/Sarimaxmodel.pkl",'wb') as file:
    pk.dump(results,file)

# Model Saved!!
# Final Model: SARIMAX(2,0,0)
# Optimizer : Powell
#
# Performance:
#     RMSE : 0.3122
#     MSE  : 0.0974
#     R²   : 0.7797
