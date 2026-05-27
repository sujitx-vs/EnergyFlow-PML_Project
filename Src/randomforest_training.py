import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
import pickle as pk

from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error

from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('Data/cleaned_data.csv')

X = data.drop('Global_active_power',axis=1)
Y = data['Global_active_power']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,random_state=42)


# Baseline model is overfitting ,so doing hyperparameter tuning.
# Training Performance -
# R2_score : 0.9790909902798902
# Testing Performance -
# R2_score : 0.8509281221382606

# params = {
#     # 'n_estimators' : [500,700,900,1000,1100], # found out 1000 is best.
#     # 'max_depth' : [33,35,37,40] # found out 35 is the best.
# }

# print("Hyper parameter tuning......")
# grid = GridSearchCV(RandomForestRegressor(n_jobs=-1,random_state=42,n_estimators=1000),params,cv=5,scoring="neg_mean_squared_error")
# grid.fit(X_train,Y_train)
# print(grid.best_params_)


model = RandomForestRegressor(n_jobs=-1,random_state=0,max_depth=35,n_estimators=1000)


model.fit(X_train,Y_train)

# training performance.
print("Training Performance -")
pred = model.predict(X_train)
mse = mean_squared_error(Y_train,pred)
rmse = root_mean_squared_error(Y_train,pred)
r2 = r2_score(Y_train,pred)
print('MSE :',mse)
print("RMSE :",rmse)
print("R2_score :",r2)

# testing performance.
print("Testing Performance -")
pred = model.predict(X_test)
mse = mean_squared_error(Y_test,pred)
rmse = root_mean_squared_error(Y_test,pred)
r2 = r2_score(Y_test,pred)
print('MSE :',mse)
print("RMSE :",rmse)
print("R2_score :",r2)


# As per Grid Search Cv we found the best params as max_depth = 35, and n_estimators = 1000. 
# with that Hyperparameters we run the model and performance of the model is as follow:
# Training Performance -
# MSE : 0.018069701101273414
# RMSE : 0.13442358833654686
# R2_score : 0.9729472957471818
# Testing Performance -
# MSE : 0.09722452764163551
# RMSE : 0.31180847910477916
# R2_score : 0.8501323722722252


# Random Forest performed reasonably well,
# but XGBoost achieved better generalization
# and superior test performance on this dataset.

# tried with some other random Hyperparameters but the r2 score decreasing to 88 in training and 83 in testing.

with open('models/RFmodel.pkl','wb') as file:
    pk.dump(model,file)


