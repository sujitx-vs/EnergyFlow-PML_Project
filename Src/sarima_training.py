import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from sklearn.metrics import root_mean_squared_error,r2_score,mean_squared_error

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX

data = pd.read_csv('Data/cleaned_data.csv')

series = data['Global_active_power']

split_index = int(series.shape[0]*.8)

series_train = series.iloc[:split_index]
series_test = series.iloc[split_index:(split_index+48)]

# since we are using time series model sarima its 
# required to check the stationarity of the data .
# to check that we use adfuller from statsmodels.
# it is Augmented Dickey-Fuller.
# in this there will be hypothesis testing going behind the scene
# and p value will be returend , and according to that p value 
# we can come to a conclusion that its stationary or not,
# if p value > 0.05 -> Non_ stationary(fail to reject Null hypothesis)
# if p value <= 0.05 -> Stationary (Rejected Null Hypothesis)
# result  = adfuller(series_train,autolag='AIC')
# p_value = result[1]

# print(p_value) # here p value is 5.566710717592664e-24, which is < 0.05
# The ADF test suggests that the series is stationary.
# so no need of differencing.

# no we want to find out the model parameters.
# for that we use ACF(Auto Correlation Function) -> visualize correlations between lags.
# and also we use PACF(Partial Auto Correlation Function) -
# to visualize partial correlations between lags.

# plot_acf(series_train)
# plot_acf(series_train, lags=45)
# plt.show()
# plot_pacf(series_train)
# plot_pacf(series_train, lags=45, method='ywm')
# plt.show()
# from these graphs we found out:

# d = 0 - Your ADF test p-value was practically 0,
# meaning the raw training series is already stationary. No differencing needed.

# p = 2 (Autoregressive order) -in PACF graph, there is a massive positive spike at lag 1,
#  a sharp negative spike at lag 2, and then at lags 3 and 4, the bars suddenly drop off.

# q = 2 (Moving Average order) - in ACF graph, There are prominent spikes at lags 1 and 2,
#  which then experience a significant drop before hit a flat pattern around lags 3 and 4.

# s = 24 : In the ACF plot, massive spikes repeatedly echo at lag 12,
#  lag 24, and lag 36.
# D = 0: The stationarity was achieved easily without any seasonal differencing.
# P = 1 (Seasonal AR): In the PACF plot, look specifically at the seasonal intervals.
#  There is a distinct, significant positive spike sitting right at lag 11/12. 
# At lag 24, that seasonal pattern completely dies off into the blue insignificance zone. 
# A sharp cutoff at the first seasonal interval means P=1.
# Q = 1 (Seasonal MA): In the ACF plot,
#  the seasonal spikes at 12, 24, and 36 stay highly significant and decay very gradually. 
# This tailing-off pattern at seasonal multiples is the classic signature of a seasonal moving average component.

# model = SARIMAX(
#     series_train,
#     order=(2, 0, 2),
#     seasonal_order=(1, 0, 1, 12),
#     enforce_stationarity=False,
#     enforce_invertibility=False
# )

# # Fit the model
# results = model.fit()
# test_steps = len(series_test)

# forecast = results.get_forecast(steps=test_steps) 
# pred = forecast.predicted_mean
# pred.index = series_test.index
# # metrices 

# rmse = root_mean_squared_error(series_test,pred)
# r2 = r2_score(series_test,pred)
# mse = mean_squared_error(series_test,pred)

# print("RMSE :",rmse)
# print("MSE :",mse)
# print("R2 Score :",r2)


# the model is trained and evaluvated but the performance way so low
# RMSE : 0.6148494798922783
# MSE : 0.37803988292380514
# R2 Score : 0.1451347171798577
# the model struggled to capture the complex consumption patterns
# effectively and produced significantly lower performance compared to XGBoost.

#checks
# print(series_train.shape)
# print(series_test.shape)


auto_model = auto_arima(
    series_train, 
    start_p=0, start_q=0,
    max_p=3, max_q=3,       # Maximum non-seasonal orders to test
    d=None,                    # Force d=0 because your data is stationary/d=None for auto_arima to find it automatically
    
    seasonal=True,          # Enable seasonal search
    m=12,                   # Your data's seasonal cycle length 
    start_P=0, start_Q=0,
    max_P=2, max_Q=2,       # Maximum seasonal orders to test
    D=None,                    # Force D=0 since seasonal differencing isn't needed/D=None for auto_arima to find it automatically.
    
    sarimax_kwargs={'low_memory':True},
    stationary=True,        # Explicitly tells the algorithm data is stationary
    stepwise=True,          # Uses a smart search algorithm (much faster than checking everything)
    trace=True,             # Prints out progress of models being tested
    error_action='ignore',  # Skips combinations that crash mathematically
    suppress_warnings=True
)

print("Best model")
print(auto_model.summary())


forecast = auto_model.predict_in_sample()

rmse = root_mean_squared_error(
    series_train,
    forecast
)

r2 = r2_score(
    series_train,
    forecast
)

print("r2 training",r2)
print("Rmse Training",rmse)


result = auto_model.predict(n_periods = len(series_test))

pred = pd.Series(result,index=series_test.index)

#checks
# print(series_test.head())
# print(pred.head())

# print(series_test.shape)
# print(pred.shape)

# print(series_test.index.equals(pred.index))

# plt.figure(figsize=(15,5))
# plt.plot(series_test[:500], label="Actual")
# plt.plot(pred[:500], label="Forecast")
# plt.legend()
# plt.show()

# print(pred.describe())
# print(series_test.describe())
# ###########################

rmse = root_mean_squared_error(series_test,pred)
r2 = r2_score(series_test,pred)
mse = mean_squared_error(series_test,pred)

print("RMSE :",rmse)
print("MSE :",mse)
print("R2 Score :",r2)


# Classical statistical time-series models were not suitable
# for this dataset. Machine Learning models such as XGBoost
# significantly outperformed SARIMA, achieving much higher
# predictive accuracy and better generalization performance.

# Multiple SARIMA and Auto-ARIMA configurations were tested.
# The best model selected by Auto-ARIMA was But still:
#
#     ARIMA(2,0,0)(2,0,0)[12]
#
# Training Performance:
#     R² Score  : 0.3340
#     RMSE      : 0.6915
#
# Testing Performance:
#     R² Score  : -0.0147
#     RMSE      : 0.8979
#     MSE       : 0.8061
#

# So This model is not saved. 

# Conclusion:
#
# Classical statistical time-series forecasting models did not perform well
# on this dataset. Multiple SARIMA and Auto-ARIMA configurations were tested,
# including seasonal and non-seasonal variants, but the models struggled to
# capture the underlying power consumption patterns effectively.
#
# The best model selected by Auto-ARIMA was:
#
#     ARIMA(2,0,0)(2,0,0)[12]
#
# Training Performance:
#     R² Score : 0.3340
#     RMSE     : 0.6915
#
# Testing Performance:
#     R² Score : -0.0147
#     RMSE     : 0.8979
#     MSE      : 0.8061
#
# The low training R² and negative testing R² indicate that the model was
# unable to explain the variance in the data and performed worse than a
# simple mean-value baseline on the test set.
#
# This suggests that the electricity consumption data contains complex
# non-linear relationships that are not captured effectively by traditional
# ARIMA-based models. In comparison, machine learning models such as
# XGBoost achieved significantly better performance and demonstrated
# stronger generalization capability.
#
# Therefore, the SARIMA/Auto-ARIMA model was not retained for deployment
# or model saving within this project.