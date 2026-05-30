import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error
# from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
# from sklearn.linear_model import RidgeCV

import pickle as pk

data = pd.read_csv('Data/cleaned_data.csv')

X = data.drop('Global_active_power',axis=1)
Y = data['Global_active_power']

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# model = LinearRegression()

# the performance of Vanila LinearRegression is not upto the mark , there is no overfitting its closer to underfitting with
# # accuracy or r2 sccore just 77. 
# Training Performance
# MSE : 0.14743528541580625
# RMSE : 0.38397302693783875
# R2 Score : 0.7792701079874215
# Testing Performance
# MSE : 0.14894014641790776
# RMSE : 0.38592764401880797
# R2 Score : 0.7704148638360636

# So tring ridge regression. 

# model = Ridge(alpha=0.1) # default value
# Ridge Regression with default alpha showed nearly identical performance to vanilla Linear Regression.


# ridge_cv = RidgeCV(alphas=[1.0,10.0,20.0])
# ridge_cv.fit(X_train,Y_train)
# v = ridge_cv.alpha_
# print(ridge_cv.alpha_) # best score is found out and it is 10.0 and used that for the Ridge still no great improvements
# the model performs
# Training Performance
# MSE : 0.14743542513532637
# RMSE : 0.383973208877034
# R2 Score : 0.7792698988090385
# Testing Performance
# MSE : 0.14893403847412137
# RMSE : 0.3859197306100342
# R2 Score : 0.770424278981272

model = Ridge(alpha=10.0)
model.fit(X_train,Y_train)

pred = model.predict(X_train)

mse = mean_squared_error(Y_train,pred)
rmse = root_mean_squared_error(Y_train,pred)
r2 = r2_score(Y_train,pred)

print("Training Performance")
print("MSE :",mse)
print("RMSE :",rmse)
print("R2 Score :",r2)


pred = model.predict(X_test)

mse = mean_squared_error(Y_test,pred)
rmse = root_mean_squared_error(Y_test,pred)
r2 = r2_score(Y_test,pred)

print("Testing Performance")
print("MSE :",mse)
print("RMSE :",rmse)
print("R2 Score :",r2)


with open("models/Linearmodel.pkl","wb") as f:
    pk.dump(model, f)

with open("models/LinearScaler.pkl","wb") as f:
    pk.dump(scaler, f)

